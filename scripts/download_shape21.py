"""Resumable, bounded HTTPS ranges for the pinned official shape checkpoint.

For connections where the Hub's whole-file transfer stalls. Only official
resolve/CDN addresses and normal TLS verification; every range and the final
SHA-256 must match before the checkpoint is made available to inference.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REVISION = '0b94677654c57bb9a6b6845cd7b704ccf551d327'
SHA = '6b519fc7242f78e9b5f47ea4d55668fe3d944a2d27332f4ca68d29a6ff603f5e'
SIZE = 7366389768
CHUNK = 4*1024*1024
URL = f'https://huggingface.co/tencent/Hunyuan3D-2.1/resolve/{REVISION}/hunyuan3d-dit-v2-1/model.fp16.ckpt'
FOLDER = ROOT/'models/hunyuan3d-2.1/hunyuan3d-dit-v2-1'


def main():
    FOLDER.mkdir(parents=True, exist_ok=True)
    output = FOLDER/'model.fp16.ckpt'
    if output.exists():
        with output.open('rb') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest() == SHA:
                print('Verified official checkpoint already present',flush=True); return
        raise ValueError('Existing checkpoint failed verification; not overwritten')
    pieces = FOLDER/'download-ranges'; pieces.mkdir(exist_ok=True)
    started = time.monotonic()
    count = (SIZE+CHUNK-1)//CHUNK

    def fetch(index):
        low, high = index*CHUNK, min(SIZE,(index+1)*CHUNK)-1
        path = pieces/f'{index:05d}.part'
        if path.exists() and path.stat().st_size == high-low+1:
            return high-low+1
        for attempt in range(4):
            try:
                request = urllib.request.Request(URL,headers={'Range':f'bytes={low}-{high}'})
                with urllib.request.urlopen(request,timeout=30) as response:
                    if response.status != 206 or response.headers.get('Content-Range') != f'bytes {low}-{high}/{SIZE}':
                        raise ValueError('Official server returned an unexpected byte range')
                    data = response.read(high-low+2)
                if len(data) != high-low+1:
                    raise ValueError('Incomplete official weight byte range')
                temporary = path.with_suffix('.tmp')
                temporary.write_bytes(data); temporary.replace(path)
                return len(data)
            except Exception:
                if attempt == 3: raise
                time.sleep(1+attempt)

    received = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(fetch,index) for index in range(count)]
        for i, future in enumerate(as_completed(futures),1):
            received += future.result()
            if i == 1 or i % 8 == 0 or i == count:
                progress = {'completed_ranges':i,'total_ranges':count,
                            'received_bytes':received,'total_bytes':SIZE,
                            'elapsed_seconds':round(time.monotonic()-started,1)}
                (FOLDER/'download-progress.json').write_text(json.dumps(progress),encoding='utf-8')
                print(json.dumps(progress),flush=True)
    temporary = output.with_suffix('.verified-download')
    digest = hashlib.sha256()
    with temporary.open('wb') as destination:
        for index in range(count):
            data = (pieces/f'{index:05d}.part').read_bytes()
            digest.update(data); destination.write(data)
    if temporary.stat().st_size != SIZE or digest.hexdigest() != SHA:
        raise ValueError('Downloaded checkpoint SHA-256 mismatch; inference file not published')
    temporary.replace(output)
    print('Official checkpoint SHA-256 verified; available for offline evaluation',flush=True)


if __name__ == '__main__':
    main()
