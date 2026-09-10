from pathlib import Path
import json
import os
import shutil
from .file_io import write_json

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
DATA.mkdir(exist_ok=True)


def engine_path(name: str):
    config = ROOT / 'engines.json'
    values = json.loads(config.read_text('utf-8')) if config.exists() else {}
    candidate = values.get(name) or os.environ.get('PHOTO3D_' + name.upper())
    if candidate and Path(candidate).is_file():
        return str(Path(candidate).resolve())
    found = shutil.which(name)
    if found:
        return found
    paths = list((ROOT / 'tools').glob('**/' + name + '.exe'))
    return str(paths[0]) if paths else None
