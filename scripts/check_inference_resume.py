"""Ensure step resumption produces the same result as uninterrupted Euler flow."""
import os
from pathlib import Path
import sys
import tempfile
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools/hunyuan3d'))
import numpy as np
import torch
from hy3dgen.shapegen.schedulers import FlowMatchEulerDiscreteScheduler, ConsistencyFlowMatchEulerDiscreteScheduler
from backend.inference_checkpoint import save_checkpoint, load_checkpoint

torch.set_num_threads(2)
sigmas = np.linspace(0, 1, 30)
def evolve(x, schedule, limit=None):
    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
    scheduler.set_timesteps(sigmas=schedule, device='cpu')
    for index, t in enumerate(scheduler.timesteps):
        prediction = x.sin() * .2 + t/1000
        x = scheduler.step(prediction, t, x).prev_sample
        if limit == index + 1:
            break
    return x

initial = torch.randn((1, 512, 64), generator=torch.Generator().manual_seed(3))
signature = {'steps': 30, 'input_sha256': 'test-image'}
with tempfile.TemporaryDirectory(dir=ROOT / 'data') as temporary:
    checkpoint = Path(temporary) / 'state.safetensors'
    partial = evolve(initial.clone(), sigmas, limit=11)
    save_checkpoint(checkpoint, partial, 11, signature)
    completed, latent = load_checkpoint(checkpoint, signature)
    resumed = evolve(latent, sigmas[completed:])
    full = evolve(initial.clone(), sigmas)
    assert torch.equal(resumed, full)
    try:
        load_checkpoint(checkpoint, {**signature, 'input_sha256': 'different-image'})
    except ValueError:
        pass
    else:
        raise AssertionError('Mismatched image checkpoint accepted')
print('INFERENCE_RESUME_OK: resumed output exactly matches uninterrupted output.')


def evolve_turbo(x, begin=0, end=5):
    scheduler = ConsistencyFlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, pcm_timesteps=100)
    scheduler.set_timesteps(num_inference_steps=5, device='cpu')
    scheduler.set_begin_index(begin)
    for t in scheduler.timesteps[begin:end]:
        prediction = (x.sin() * .2 + t.to(x.dtype)/1000).to(x.dtype)
        x = scheduler.step(prediction, t, x).prev_sample
    return x


initial = torch.randn((1, 3072, 64), generator=torch.Generator().manual_seed(3), dtype=torch.float16)
signature = {'steps': 5, 'input_sha256': 'test-image', 'num_latents': 3072, 'dtype': 'float16',
             'scheduler': 'ConsistencyFlowMatchEulerDiscreteScheduler'}
with tempfile.TemporaryDirectory(dir=ROOT / 'data') as temporary:
    checkpoint = Path(temporary) / 'state.safetensors'
    partial = evolve_turbo(initial.clone(), end=2)
    save_checkpoint(checkpoint, partial, 2, signature)
    completed, latent = load_checkpoint(checkpoint, signature)
    resumed = evolve_turbo(latent, begin=completed)
    assert torch.equal(resumed, evolve_turbo(initial.clone()))
    for change in ({'input_sha256': 'other'}, {'dtype': 'float32'}, {'num_latents': 512}):
        try:
            load_checkpoint(checkpoint, {**signature, **change})
        except ValueError:
            pass
        else:
            raise AssertionError('Mismatched Turbo checkpoint accepted')
print('TURBO_RESUME_OK: FP16 3072-latent resume exactly matches the full 5-step schedule.')
