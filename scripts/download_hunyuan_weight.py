"""Resumable download from Tencent's ModelScope repo, checked against both hubs."""
import hashlib
import argparse
import json
from pathlib import Path
import re
import shutil
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'models/hunyuan3d-2mini/hunyuan3d-dit-v2-mini/model.fp16.safetensors'
PART = DEST.with_suffix('.safetensors.download')
SHA = '3cc66f3bea33e4062b7dbc875ffe1d70c4888914aec3e91b60f94e9bd01b522b'
SIZE = 3819958234
REVISION = '0d1901d5f6300a268615cff3aff885c4fd494ff5'
URL = ('https://modelscope.cn/api/v1/models/Tencent-Hunyuan/Hunyuan3D-2mini/repo'
       f'?Revision={REVISION}&FilePath=hunyuan3d-dit-v2-mini/model.fp16.safetensors')
HF_REVISION = 'f90a0f7df7d5e6f71109cf333f6a95a0ae3194a6'
HF_URL = (f'https://huggingface.co/tencent/Hunyuan3D-2mini/resolve/{HF_REVISION}/'
          'hunyuan3d-dit-v2-mini/model.fp16.safetensors?download=true')
PROVIDER = 'ModelScope'
STATUS = ROOT / 'data/weight-download.json'


def report(state, size, **extra):
    STATUS.parent.mkdir(exist_ok=True)
    temporary = STATUS.with_suffix('.tmp')
    temporary.write_text(json.dumps({'state': state, 'bytes': size, 'total_bytes': SIZE,
        'source': 'Tencent-Hunyuan/Hunyuan3D-2mini on ' + PROVIDER,
        'updated_at': time.time(), **extra}, indent=2), 'utf-8')
    temporary.replace(STATUS)


def main():
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if not PART.exists() and not DEST.exists():
        previous = ROOT / 'models/hunyuan3d-2mini/.cache/huggingface/download/hunyuan3d-dit-v2-mini'
        candidates = list(previous.glob('*.'+SHA+'.incomplete'))
        if candidates:
            shutil.copy2(candidates[0], PART)
    offset = DEST.stat().st_size if DEST.exists() else PART.stat().st_size if PART.exists() else 0
    if DEST.exists() and offset != SIZE:
        raise RuntimeError('Installed file has an unexpected size; review it before downloading.')
    if offset > SIZE:
        raise RuntimeError('Existing download is larger than the official file; review it manually.')
    failures = 0
    report('starting', offset)
    print(f'Resuming official weight at {offset} / {SIZE} bytes.', flush=True)
    while offset < SIZE:
        end = min(offset + 16*1024*1024, SIZE) - 1
        request = urllib.request.Request(URL, headers={'Range': f'bytes={offset}-{end}'})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
                if not match or tuple(map(int, match.groups())) != (offset, end, SIZE):
                    raise RuntimeError('Server did not honor the exact requested byte range.')
                remaining = end - offset + 1
                reader = getattr(response, 'read1', response.read)
                window_started = last_report = time.monotonic()
                window_bytes = 0
                with PART.open('ab') as stream:
                    while remaining:
                        chunk = reader(min(64*1024, remaining))
                        if not chunk:
                            raise IOError('Download connection ended early.')
                        stream.write(chunk)
                        offset += len(chunk)
                        remaining -= len(chunk)
                        window_bytes += len(chunk)
                        now = time.monotonic()
                        if now - last_report >= 5:
                            stream.flush()
                            report('downloading', offset)
                            last_report = now
                        if now - window_started >= 30:
                            if window_bytes < 128*1024:
                                raise IOError('Transfer stalled; reconnecting with the saved offset.')
                            window_started, window_bytes = now, 0
            failures = 0
            report('downloading', offset)
            print(f'{offset / SIZE:.1%} ({offset // 1048576} / {SIZE // 1048576} MiB)', flush=True)
        except (OSError, urllib.error.URLError) as error:
            failures += 1
            offset = PART.stat().st_size if PART.exists() else 0
            report('retrying', offset, error=str(error))
            if failures >= 4:
                report('failed', offset, error=str(error))
                raise
            delay = 5 * failures
            if isinstance(error, urllib.error.HTTPError) and error.code == 429:
                try:
                    delay = max(delay, int(error.headers.get('Retry-After', '30')))
                except ValueError:
                    delay = 60
                if delay > 60:
                    report('failed', offset, error='Server requested a longer cooldown; resume later.')
                    raise
            print(f'Transfer interrupted; resuming after {delay}s.', flush=True)
            time.sleep(delay)
    target = DEST if DEST.exists() else PART
    report('verifying', offset)
    with target.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    if sha != SHA:
        report('failed', offset, error='SHA-256 mismatch; file will not be loaded.')
        raise RuntimeError('SHA-256 mismatch; preserving file for inspection.')
    if target == PART:
        PART.replace(DEST)
    report('verified', offset, sha256=sha, revision=REVISION)
    print('Official weight SHA-256 verified.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=['modelscope', 'huggingface'], default='modelscope')
    args = parser.parse_args()
    if args.source == 'huggingface':
        URL, REVISION, PROVIDER = HF_URL, HF_REVISION, 'Hugging Face'
    main()
