"""Compare the CPU adapter with direct official VAE loading and dense sampling."""
import os
from pathlib import Path
import sys
import argparse
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HY3DGEN_DEBUG='0', OMP_NUM_THREADS='8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools/hunyuan3d'))
sys.path.insert(0, str(ROOT / 'backend'))
import json
import numpy as np
import torch
import yaml
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from hy3dgen.shapegen.pipelines import instantiate_from_config, export_to_trimesh
from hy3dgen.shapegen.models.autoencoders import VanillaVolumeDecoder, MCSurfaceExtractor

torch.set_num_threads(8)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--latent', default=str(ROOT / 'data/cat-2mini-validation-run2/shape_latents.safetensors'))
parser.add_argument('--output', default=str(ROOT / 'data/cat-2mini-validation-run2'))
parser.add_argument('--resolution', type=int, default=63)
parser.add_argument('--adaptive', action='store_true')
parser.add_argument('--dtype', choices=['float32', 'float16'], default='float32')
parser.add_argument('--math-attention', action='store_true')
parser.add_argument('--device', choices=['cpu', 'xpu'], default='cpu')
parser.add_argument('--variant', choices=['standard', 'turbo'], default='standard')
args = parser.parse_args()
if args.device == 'xpu':
    from bounded_attention import install
    install()
folder = ROOT / 'models/hunyuan3d-2mini' / ('hunyuan3d-dit-v2-mini-turbo' if args.variant == 'turbo' else 'hunyuan3d-dit-v2-mini')
output = Path(args.output)
output.mkdir(exist_ok=True)
config = yaml.safe_load((folder / 'config.yaml').read_text('utf-8'))
vae = instantiate_from_config(config['vae'])
del vae.encoder
del vae.pre_kl
with safe_open(folder / 'model.fp16.safetensors', framework='pt', device='cpu') as handle:
    state = {k[4:]: handle.get_tensor(k) for k in handle.keys()
             if k.startswith('vae.') and not k.startswith(('vae.encoder.', 'vae.pre_kl.'))}
vae.load_state_dict(state, strict=True)
del state
vae.eval().requires_grad_(False)
latent = load_file(args.latent)['latents']
dtype = getattr(torch, args.dtype)
vae.to(device=args.device, dtype=dtype)
latent = latent.to(device=args.device, dtype=dtype)
if args.math_attention:
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
print('Frequencies', vae.fourier_embedder.frequencies.tolist(), flush=True)
print('Latent', latent.mean().item(), latent.std().item(), flush=True)
with torch.inference_mode():
    decoded = vae(latent / vae.scale_factor)
    print('Decoded', decoded.mean().item(), decoded.std().item(), flush=True)
    save_file({'decoded': decoded.cpu().contiguous()}, str(output / 'diagnostic_decoded.safetensors'))
    from cpu_volume import CPUVolumeDecoder
    decoder = CPUVolumeDecoder(lambda *a, **kw: print(a, kw, flush=True)) if args.adaptive else VanillaVolumeDecoder()
    grid = decoder(decoded, vae.geo_decoder, bounds=1.01, octree_resolution=args.resolution, num_chunks=2048)
    positive = (grid[0] > 0).nonzero()
    print('Positive count/bounds', len(positive), positive.min(0).values.tolist() if len(positive) else [],
          positive.max(0).values.tolist() if len(positive) else [], flush=True)
    np.save(output / 'diagnostic_dense_grid.npy', grid.float().cpu().numpy())
    meshes = MCSurfaceExtractor()(grid, bounds=1.01, octree_resolution=args.resolution, mc_level=0)
    mesh = export_to_trimesh(meshes)[0]
    if mesh is None:
        raise SystemExit('No zero crossing: the neural field did not produce a closed object at this resolution.')
    mesh.export(output / 'diagnostic_dense.ply')
    print('Dense mesh', len(mesh.faces), mesh.extents.tolist(), flush=True)
