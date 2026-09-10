"""Atomic, image-bound checkpoints for deterministic Euler shape inference."""
import json
from pathlib import Path
import torch
import uuid
from safetensors import safe_open
from safetensors.torch import save_file
if __package__:
    from .file_io import replace_with_retry
else:
    from file_io import replace_with_retry


def save_checkpoint(path, latent, completed, signature):
    path = Path(path)
    temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        save_file({'latents': latent.detach().cpu().contiguous()}, str(temporary),
                  metadata={'signature': json.dumps(signature, sort_keys=True), 'completed': str(completed)})
        replace_with_retry(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(path, signature):
    path = Path(path)
    if path.stat().st_size > 1024 * 1024:
        raise ValueError('Unexpected checkpoint size.')
    with safe_open(path, framework='pt', device='cpu') as handle:
        meta = handle.metadata() or {}
        if json.loads(meta.get('signature', '{}')) != signature:
            raise ValueError('断点与照片或生成参数不匹配，不能混用。')
        completed = int(meta.get('completed', '-1'))
        latent = handle.get_tensor('latents')
    count = signature.get('num_latents', 512)
    if count not in (512, 1024, 2048, 3072):
        raise ValueError('Unexpected latent count.')
    if not 0 < completed <= signature['steps'] or tuple(latent.shape) != (1, count, 64):
        raise ValueError('Unexpected checkpoint state.')
    expected_dtype = {'float32': torch.float32, 'float16': torch.float16}.get(signature.get('dtype', 'float32'))
    if latent.dtype != expected_dtype or not torch.isfinite(latent).all():
        raise ValueError('Invalid checkpoint tensor.')
    return completed, latent
