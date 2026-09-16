"""Per-machine paths. Configuration is local and never accepted through HTTP."""
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
WEIGHT_SHA = 'bdbcef30dd0149a281e17d5b5b1fdad1122c904e098a42f3100e04e03c247bc4'
CONFIG_SHA = 'be28205844da01bd5d3c5ba5160f5886fb9765d542483f9d12bda8f17324db5e'
CODE_REVISION = 'f8db63096c8282cb27354314d896feba5ba6ff8a'
MODEL_REVISION = 'f90a0f7df7d5e6f71109cf333f6a95a0ae3194a6'


def config():
    file = ROOT / 'photoform.json'
    values = json.loads(file.read_text('utf-8-sig')) if file.exists() else {}
    def path(key, default):
        p = Path(values.get(key) or default).expanduser()
        return (p if p.is_absolute() else ROOT / p).resolve()
    local_env = ROOT / '.ai-env'
    legacy = ROOT.parent / 'photo-to-print-xpu-conda'
    env = path('ai_env', legacy if (legacy/'python.exe').exists() and not (local_env/'python.exe').exists() else local_env)
    conda = values.get('conda') or os.environ.get('CONDA_EXE') or shutil.which('conda')
    if not conda:
        candidates = [Path.home()/'miniconda3/Scripts/conda.exe', Path('C:/miniconda3/Scripts/conda.exe')]
        conda = next((str(p) for p in candidates if p.is_file()), 'conda')
    return {'ai_env': env, 'conda': os.path.normcase(str(conda)),
            'model_dir': path('model_dir', 'models/hunyuan3d-2mini/hunyuan3d-dit-v2-mini-turbo'),
            'source_dir': path('source_dir', 'tools/hunyuan3d'),
            'rembg_dir': path('rembg_dir', 'models/rembg')}


def fingerprint(values=None):
    c = values or config()
    paths = [c['ai_env']/'python.exe', c['ai_env']/'Lib/site-packages/torch/__init__.py',
             c['model_dir']/'model.fp16.safetensors', c['model_dir']/'config.yaml',
             c['source_dir']/'hy3dgen/shapegen/pipelines.py', c['rembg_dir']/'u2net.onnx']
    result = {'paths': {key: str(value) for key, value in c.items()}, 'files': []}
    for path in paths:
        info = path.stat() if path.is_file() else None
        result['files'].append({'path': str(path), 'size': info.st_size if info else None,
                                'mtime_ns': info.st_mtime_ns if info else None})
    return result
