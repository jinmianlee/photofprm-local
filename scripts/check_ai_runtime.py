"""Validate pinned architecture against tensor metadata, without loading weights."""
import json
import os
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HY3DGEN_DEBUG='0')
sys.path.insert(0, str(ROOT / 'tools' / 'hunyuan3d'))
import torch
import yaml
from accelerate import init_empty_weights
from hy3dgen.shapegen.pipelines import instantiate_from_config

torch.set_num_threads(4)
folder = ROOT / 'models' / 'hunyuan3d-2mini' / 'hunyuan3d-dit-v2-mini'
weight = folder / 'model.fp16.safetensors'
if not weight.exists():
    weight = weight.with_suffix('.safetensors.download')
with weight.open('rb') as stream:
    length = struct.unpack('<Q', stream.read(8))[0]
    assert 0 < length < 16 * 1024 * 1024
    tensors = json.loads(stream.read(length))
config = yaml.safe_load((folder / 'config.yaml').read_text('utf-8'))
result = {}
for name in ('model', 'vae', 'conditioner'):
    with init_empty_weights(include_buffers=False):
        module = instantiate_from_config(config[name])
    if name == 'vae':
        del module.encoder
        del module.pre_kl
    expected = {k: list(v.shape) for k, v in module.state_dict().items()}
    actual = {k[len(name)+1:]: v['shape'] for k, v in tensors.items()
              if k.startswith(name + '.')}
    if name == 'vae':
        actual = {k: v for k, v in actual.items() if not k.startswith(('encoder.', 'pre_kl.'))}
    missing = sorted(expected.keys() - actual.keys())
    extra = sorted(actual.keys() - expected.keys())
    wrong = [k for k in expected.keys() & actual.keys() if expected[k] != actual[k]]
    result[name] = {'parameters': sum(p.numel() for p in module.parameters()),
                    'tensors': len(actual), 'missing': missing, 'extra': extra, 'shape_mismatch': wrong}
    print(name, result[name], flush=True)
    assert not missing and not extra and not wrong
    del module
report = {'architecture_validated': True, 'weight_integrity_validated': False,
          'inference_validated': False, 'torch': torch.__version__, 'components': result}
(ROOT / 'data' / 'ai-architecture-check.json').write_text(json.dumps(report, indent=2), 'utf-8')
print('AI_ARCHITECTURE_OK; full file integrity and inference still require validation.', flush=True)
