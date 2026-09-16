"""Reproducible Windows / Intel XPU setup from official sources. No driver changes."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.runtime_config import config, MODEL_REVISION


def plan(c):
    prefix = [str(c['conda']), 'run', '--prefix', str(c['ai_env']), '--no-capture-output']
    return {
        'dependencies': [
            [str(c['conda']), 'create', '--prefix', str(c['ai_env']), '--override-channels', '-c',
             'https://repo.anaconda.com/pkgs/main', 'python=3.12', 'pip', '-y'],
            [*prefix, 'python', '-m', 'pip', 'install', '--index-url', 'https://download.pytorch.org/whl/xpu',
             'torch==2.6.0+xpu', 'torchvision==0.21.0+xpu'],
            [*prefix, 'python', '-m', 'pip', 'install', '--index-url', 'https://pypi.org/simple',
             '-r', str(ROOT/'requirements-xpu-installed.txt')],
            [*prefix, 'python', '-m', 'pip', 'check']],
        'models': [[*prefix, 'hf', 'download', 'tencent/Hunyuan3D-2mini',
                    'LICENSE', 'NOTICE', 'README.md',
                    'hunyuan3d-dit-v2-mini-turbo/config.yaml',
                    'hunyuan3d-dit-v2-mini-turbo/model.fp16.safetensors',
                    '--revision', MODEL_REVISION, '--local-dir', str(c['model_dir'].parent), '--max-workers', '1'],
                   [*prefix, 'python', str(ROOT/'backend/foreground.py'), '--install']],
        'check': [[sys.executable, str(ROOT/'scripts/configure_engine.py')]],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['all', 'dependencies', 'source', 'models', 'check'], default='all')
    parser.add_argument('--dry-run', action='store_true', help='Show commands without downloads or changes')
    args = parser.parse_args()
    c = config()
    if c['model_dir'].name != 'hunyuan3d-dit-v2-mini-turbo':
        raise SystemExit('For downloading, model_dir must end with hunyuan3d-dit-v2-mini-turbo. Existing custom directories can use configure_engine.py.')
    stages = ['dependencies', 'source', 'models', 'check'] if args.stage == 'all' else [args.stage]
    commands = plan(c)
    environment = os.environ.copy()
    environment.update(PYTHONUTF8='1', HF_HUB_DISABLE_TELEMETRY='1', HF_HUB_DISABLE_IMPLICIT_TOKEN='1',
                       HF_HUB_DISABLE_XET='1', HF_HUB_DOWNLOAD_TIMEOUT='120', PIP_DISABLE_PIP_VERSION_CHECK='1')
    for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE'):
        environment.pop(key, None)
    for stage in stages:
        print('Stage: '+stage, flush=True)
        if stage == 'source':
            print('Official Tencent-Hunyuan/Hunyuan3D-2 source at the pinned revision.', flush=True)
            if not args.dry_run:
                import prepare_single_photo
                prepare_single_photo.SOURCE = c['source_dir']
                prepare_single_photo.source()
            continue
        for index, command in enumerate(commands[stage]):
            if stage == 'dependencies' and index == 0 and (c['ai_env']/'python.exe').is_file():
                continue
            print(subprocess.list2cmdline(command), flush=True)
            if not args.dry_run:
                subprocess.run(command, cwd=ROOT, env=environment, check=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
