"""Connect existing local model/runtime paths and run an actual environment check."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.runtime_config import config
from backend.file_io import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('ai-env', 'conda', 'model-dir', 'source-dir', 'rembg-dir'):
        parser.add_argument('--'+key)
    parser.add_argument('--no-check', action='store_true', help='Save paths only; generation remains disabled until the check passes')
    args = parser.parse_args()
    existing = ROOT/'photoform.json'
    values = json.loads(existing.read_text('utf-8-sig')) if existing.exists() else {}
    changes = {key: value for key, value in vars(args).items() if key != 'no_check' and value is not None}
    if changes:
        values.update(changes)
        write_json(existing, values)
    if args.no_check:
        return 0
    c = config()
    environment = os.environ.copy()
    environment.update(PYTHONUTF8='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    result = subprocess.run([str(c['conda']), 'run', '--prefix', str(c['ai_env']), '--no-capture-output',
                             'python', str(ROOT/'scripts/verify_engine.py')], cwd=ROOT, env=environment)
    if result.returncode:
        print('Check failed. Read data/engine-check.json and DEPLOYMENT.md; no quality validation was fabricated.')
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
