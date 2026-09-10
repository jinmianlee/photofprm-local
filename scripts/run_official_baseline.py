"""Diagnostic baseline using upstream loading and sampling, entirely offline."""
import os
from pathlib import Path
import sys
import time
import json
import argparse
ROOT = Path(__file__).resolve().parents[1]
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HY3DGEN_DEBUG='0', OMP_NUM_THREADS='8')
sys.path.insert(0, str(ROOT / 'tools/hunyuan3d'))
sys.path.insert(0, str(ROOT / 'backend'))
import torch
from PIL import Image
from safetensors.torch import save_file
from hy3dgen.shapegen.pipelines import Hunyuan3DDiTFlowMatchingPipeline
from inference_checkpoint import save_checkpoint
import hashlib

torch.set_num_threads(8)
torch.set_num_interop_threads(1)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--seed', type=int, default=12345)
parser.add_argument('--steps', type=int, default=30)
parser.add_argument('--output', default=str(ROOT / 'data/cat-official-baseline'))
parser.add_argument('--dtype', choices=['float32', 'float16'], default='float32')
parser.add_argument('--input', default=str(ROOT / 'data/cat-2mini-validation/foreground.png'))
parser.add_argument('--device', choices=['cpu', 'xpu'], default='cpu')
parser.add_argument('--guidance', type=float, default=5.0)
parser.add_argument('--num-latents', type=int, choices=[512, 1024, 2048, 3072])
parser.add_argument('--variant', choices=['standard', 'turbo'], default='standard')
args = parser.parse_args()
if args.device == 'xpu':
    from bounded_attention import install
    install()
folder = ROOT / 'models/hunyuan3d-2mini' / ('hunyuan3d-dit-v2-mini-turbo' if args.variant == 'turbo' else 'hunyuan3d-dit-v2-mini')
output = Path(args.output)
output.mkdir(exist_ok=True)
image_path = Path(args.input)
started = time.monotonic()
signature = {'input_sha256': hashlib.sha256(image_path.read_bytes()).hexdigest(),
             'weight_sha256': ('bdbcef30dd0149a281e17d5b5b1fdad1122c904e098a42f3100e04e03c247bc4' if args.variant == 'turbo' else '3cc66f3bea33e4062b7dbc875ffe1d70c4888914aec3e91b60f94e9bd01b522b'),
             'steps': args.steps, 'seed': args.seed, 'guidance': args.guidance, 'dtype': args.dtype}
pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_single_file(
    str(folder / 'model.fp16.safetensors'), str(folder / 'config.yaml'),
    device=args.device, dtype=getattr(torch, args.dtype), use_safetensors=True)
if args.num_latents:
    pipeline.vae.latent_shape = (args.num_latents, pipeline.vae.latent_shape[1])
signature['num_latents'] = pipeline.vae.latent_shape[0]
(output / 'settings.json').write_text(json.dumps(signature, indent=2), 'utf-8')
def progress(index, t, state):
    save_checkpoint(output / 'diffusion_checkpoint.safetensors', state.prev_sample, index+1, signature)
    report = {'step': index+1, 'total': args.steps, 'elapsed_seconds': round(time.monotonic()-started, 2)}
    (output / 'progress.json').write_text(json.dumps(report), 'utf-8')
    print(report, flush=True)
latent = pipeline(image=Image.open(image_path), num_inference_steps=args.steps,
                  guidance_scale=args.guidance,
                  generator=torch.Generator().manual_seed(args.seed),
                  callback=progress, callback_steps=1, output_type='latent')
save_file({'latents': latent.detach().cpu().contiguous()}, str(output / 'shape_latents.safetensors'))
print('OFFICIAL_BASELINE_LATENT_SAVED', flush=True)
