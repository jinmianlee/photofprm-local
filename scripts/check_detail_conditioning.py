"""Check ROI-conditioning invariants without loading any model weights."""
from pathlib import Path
import sys
import tempfile
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.detail_conditioning import enrich_conditioning


class EncodedCrop:
    calls = 0
    def encode_cond(self, image, **kwargs):
        self.calls += 1
        assert image.shape == (1, 3, 64, 64)
        return {'main': torch.tensor([1., -1., 2.]).repeat(1, 65, 1)}


pipeline = EncodedCrop()
image = torch.zeros(1, 3, 64, 64)
original = torch.tensor([2., 1., -1.]).repeat(1, 65, 1)
condition = {'main': original.clone()}
mixed, report = enrich_conditioning(pipeline, image, condition, [.25, .25, .75, .75], .35)
assert torch.equal(mixed['main'][:, :1], original[:, :1])
ys, xs = torch.meshgrid(torch.arange(8), torch.arange(8), indexing='ij')
outside = ((xs < 2) | (xs >= 6) | (ys < 2) | (ys >= 6)).reshape(-1)
assert torch.equal(mixed['main'][:, 1:, :][:, outside], original[:, 1:, :][:, outside])
assert not torch.equal(mixed['main'][:, 1:, :][:, ~outside], original[:, 1:, :][:, ~outside])
assert torch.equal(condition['main'], original), 'Caller input was mutated'
assert torch.isfinite(mixed['main']).all()
unchanged, _ = enrich_conditioning(pipeline, image, condition, [.25, .25, .75, .75], 0)
assert unchanged is condition and pipeline.calls == 1
for roi, strength in [([float('nan'), 0, 1, 1], .3), ([0, 0, 1.1, 1], .3), ([.5, .5, .51, .51], .3), ([0, 0, 1, 1], .7)]:
    try:
        enrich_conditioning(pipeline, image, condition, roi, strength)
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid experimental ROI accepted')
print('DETAIL_CONDITIONING_INVARIANTS_OK: global and non-ROI tokens unchanged; bounded local changes only.')

from backend.inference_checkpoint import save_checkpoint, load_checkpoint
signature = {'steps':5,'num_latents':3072,'dtype':'float16','input_sha256':'same-image'}
adapted = {**signature,'conditioning_adapter':{'method':'multiscale_dino_roi_v1','roi':[.25,.25,.75,.75],'strength':.35}}
with tempfile.TemporaryDirectory(dir=ROOT/'data') as folder:
    checkpoint = Path(folder)/'checkpoint.safetensors'
    save_checkpoint(checkpoint, torch.zeros(1,3072,64,dtype=torch.float16), 5, adapted)
    assert load_checkpoint(checkpoint, adapted)[0] == 5
    try:
        load_checkpoint(checkpoint, signature)
    except ValueError:
        pass
    else:
        raise AssertionError('Experimental checkpoint accepted by the default generation signature')
print('DETAIL_CHECKPOINT_ISOLATION_OK: experimental and default checkpoints cannot be mixed.')
