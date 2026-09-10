"""Download official dependencies/code/weights. This does not certify inference.

Run with the existing app runtime. All new files stay in the workspace.
HTTPS verification remains enabled; a failed stage stops with a resumable log.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import shutil

ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT.parent / 'photo-to-print-ai-conda'
PYTHON = ENV / ('python.exe' if os.name == 'nt' else 'bin/python')
HF = ENV / ('Scripts/hf.exe' if os.name == 'nt' else 'bin/hf')
CONDA = shutil.which('conda') or 'C:/miniconda3/Scripts/conda.exe'
SOURCE = ROOT / 'tools' / 'hunyuan3d'
MODEL = ROOT / 'models' / 'hunyuan3d-2mini'
CODE_REVISION = 'f8db63096c8282cb27354314d896feba5ba6ff8a'
MODEL_REVISION = 'f90a0f7df7d5e6f71109cf333f6a95a0ae3194a6'
REPO = 'tencent/Hunyuan3D-2mini'
FILES = ['LICENSE', 'NOTICE', 'README.md',
         'hunyuan3d-dit-v2-mini/config.yaml',
         'hunyuan3d-dit-v2-mini/model.fp16.safetensors']
STATUS = ROOT / 'data' / 'single-photo-install.json'
OFFICIAL_FILES = {
    'hunyuan3d-dit-v2-mini/config.yaml': (1628, 'cabcba7f6115752c8fe5b370e12bf714936f70377a8a80f151872f76c2d64609'),
    'hunyuan3d-dit-v2-mini/model.fp16.safetensors': (3819958234, '3cc66f3bea33e4062b7dbc875ffe1d70c4888914aec3e91b60f94e9bd01b522b'),
}


def conda_python(*args):
    return [CONDA, 'run', '--prefix', ENV, '--no-capture-output', 'python', *args]


def record(stage, state, message):
    STATUS.parent.mkdir(exist_ok=True)
    value = {'stage': stage, 'state': state, 'message': message,
             'updated_at': time.time(), 'inference_validated': False,
             'code_revision': CODE_REVISION, 'model_revision': MODEL_REVISION}
    temp = STATUS.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), 'utf-8')
    temp.replace(STATUS)


def run(args, cwd=ROOT):
    print('Running:', ' '.join(str(x) for x in args), flush=True)
    environment = os.environ.copy()
    environment.update(HF_HUB_DISABLE_TELEMETRY='1', HF_HUB_DISABLE_XET='1',
                       HF_HUB_DISABLE_IMPLICIT_TOKEN='1', HF_HUB_DOWNLOAD_TIMEOUT='60',
                       HF_HOME=str(ROOT / 'models' / '.hf-cache'),
                       CONDA_PKGS_DIRS=str(ROOT.parent / '.conda-packages'),
                       PIP_DISABLE_PIP_VERSION_CHECK='1', PYTHONUTF8='1')
    subprocess.run([str(x) for x in args], cwd=cwd, env=environment, check=True)


def dependencies():
    if not PYTHON.is_file():
        run([CONDA, 'create', '--prefix', ENV, '--override-channels', '-c',
             'https://repo.anaconda.com/pkgs/main', 'python=3.12', 'pip', '-y', '--quiet'])
    prefix = conda_python()
    run([*prefix, '-m', 'pip', 'install', '--index-url',
         'https://download.pytorch.org/whl/cpu', '--timeout', '30',
         'torch==2.10.0', 'torchvision==0.25.0'])
    run([*prefix, '-m', 'pip', 'install', '--index-url', 'https://pypi.org/simple',
         '--timeout', '30', '-r', ROOT / 'requirements-ai.txt'])
    run([*prefix, '-m', 'pip', 'check'])
    result = subprocess.check_output([str(x) for x in conda_python('-m', 'pip', 'freeze')], text=True)
    (ROOT / 'requirements-ai-installed.txt').write_text(result, 'utf-8')


def source():
    if SOURCE.exists() and not (SOURCE / '.git').is_dir():
        raise RuntimeError(f'Refusing to overwrite a non-Git directory: {SOURCE}')
    if not SOURCE.exists():
        SOURCE.parent.mkdir(exist_ok=True)
        run(['git', 'clone', '--depth', '1', '--filter=blob:none', '--sparse',
             'https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git', SOURCE])
    origin = subprocess.check_output(['git', '-C', str(SOURCE), 'remote', 'get-url', 'origin'], text=True).strip()
    if origin != 'https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git':
        raise RuntimeError('Existing source directory does not have the expected official origin.')
    dirty = subprocess.check_output(['git', '-C', str(SOURCE), 'status', '--porcelain'], text=True)
    if dirty.strip():
        raise RuntimeError('Source has local edits; preserve them and review before updating.')
    present = subprocess.run(['git', '-C', str(SOURCE), 'cat-file', '-e',
                              CODE_REVISION + '^{commit}'], capture_output=True)
    if present.returncode:
        run(['git', '-C', SOURCE, 'fetch', '--depth', '1', 'origin', CODE_REVISION])
    run(['git', '-C', SOURCE, 'checkout', '--detach', CODE_REVISION])
    run(['git', '-C', SOURCE, 'sparse-checkout', 'set', 'hy3dgen/shapegen'])
    head = subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip()
    if head != CODE_REVISION:
        raise RuntimeError('Unexpected source revision.')


def weights():
    if not HF.is_file():
        raise RuntimeError('Install the Hugging Face CLI in the AI runtime first.')
    small_files = [name for name in FILES if not name.endswith('.safetensors')]
    run([CONDA, 'run', '--prefix', ENV, '--no-capture-output', HF,
         'download', REPO, *small_files, '--revision', MODEL_REVISION,
         '--local-dir', MODEL, '--max-workers', '1'])
    run(conda_python(ROOT / 'scripts' / 'download_hunyuan_weight.py'))
    verify()


def foreground():
    run(conda_python(ROOT / 'backend' / 'foreground.py', '--install'))


def verify():
    manifest = []
    for name in FILES:
        path = MODEL / name
        if not path.is_file():
            raise RuntimeError(f'Missing file: {name}')
        sha = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                sha.update(chunk)
        entry = {'file': name, 'bytes': path.stat().st_size, 'sha256': sha.hexdigest()}
        if name in OFFICIAL_FILES:
            expected_size, expected_sha = OFFICIAL_FILES[name]
            if sha.hexdigest() != expected_sha or path.stat().st_size != expected_size:
                raise RuntimeError(f'File does not match pinned official size/SHA-256: {name}')
            entry['verified_against_official_sha256'] = True
        manifest.append(entry)
    (MODEL / 'download-manifest.json').write_text(json.dumps({
        'repo': REPO, 'revision': MODEL_REVISION, 'files': manifest,
        'weight_sources': {'huggingface': MODEL_REVISION,
                           'modelscope': '0d1901d5f6300a268615cff3aff885c4fd494ff5'},
        'metadata_checked_at': '2026-09-06',
        'inference_validated': False}, indent=2), 'utf-8')
    print('Files verified. CPU inference and cat quality still require validation.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['all', 'dependencies', 'source', 'weights', 'foreground', 'verify'], default='all')
    args = parser.parse_args()
    stages = ['dependencies', 'source', 'weights', 'foreground'] if args.stage == 'all' else [args.stage]
    for stage in stages:
        record(stage, 'running', 'Preparing official files; inference is not validated.')
        try:
            globals()[stage]()
        except (Exception, KeyboardInterrupt) as error:
            record(stage, 'failed', f'{type(error).__name__}: {error}')
            print(f'Stopped at {stage}: {error}', file=sys.stderr)
            return 1
        record(stage, 'prepared', 'Stage prepared; CPU inference is not validated.')
    return 0


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    raise SystemExit(main())
