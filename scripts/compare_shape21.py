"""Evaluate pinned official Hunyuan3D-2.1 shape weights on Intel XPU, offline.

Experimental comparison, not the default engine or a likeness guarantee.
Stage-wise loading and exact query-chunked attention bound device allocations.
The official checkpoint is hash-verified and opened with weights_only=True;
no arbitrary pickle globals, network model loading or custom CUDA extensions.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))
from file_io import write_json

SOURCE_REVISION = '82920d643c0dc2f7bfd7255f45f62d386edfe60c'
MODEL_REVISION = '0b94677654c57bb9a6b6845cd7b704ccf551d327'
WEIGHT_SHA = '6b519fc7242f78e9b5f47ea4d55668fe3d944a2d27332f4ca68d29a6ff603f5e'
CONFIG_SHA = 'a9e9b66f0163a9b827d730633ac88f47fcc8e3071dcbb9ee12d88ef7537c5c6e'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--source', default=str(ROOT/'tools/Hunyuan3D-2.1'))
    parser.add_argument('--model', default=str(ROOT/'models/hunyuan3d-2.1/hunyuan3d-dit-v2-1'))
    parser.add_argument('--steps', type=int, choices=[5, 30, 50], default=50)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--resolution', type=int, choices=[255, 383], default=255)
    parser.add_argument('--batched-cfg', action='store_true',
                        help='Diagnostic: match the official two-sample CFG forward exactly; uses more memory')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    source, folder = Path(args.source).resolve(), Path(args.model).resolve()
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    metrics = {'model': 'Hunyuan3D-Shape-2.1', 'model_revision': MODEL_REVISION,
               'source_revision': SOURCE_REVISION, 'state': 'running',
               'steps': args.steps, 'seed': args.seed, 'resolution': args.resolution,
               'tokens': 4096, 'cfg': 5., 'cfg_mode': 'batched' if args.batched_cfg else 'sequential',
               'experimental': True, 'likeness_validated': False}

    def report(stage, **fields):
        metrics.update(stage=stage, elapsed_seconds=round(time.monotonic()-started, 2), **fields)
        write_json(output/'inference.json', metrics)
        print(json.dumps({'stage': stage, **fields}, ensure_ascii=False), flush=True)

    try:
        report('Verify pinned official shape source and files')
        revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
        if revision != SOURCE_REVISION:
            raise ValueError('Official source revision mismatch')
        if subprocess.check_output(['git', '-C', str(source), 'diff', 'HEAD', '--', 'hy3dshape/hy3dshape'], text=True):
            raise ValueError('Official shape source has local changes')
        if subprocess.check_output(['git', '-C', str(source), 'ls-files', '--others', '--exclude-standard', '--', 'hy3dshape/hy3dshape'], text=True).strip():
            raise ValueError('Unexpected files in official shape package')
        weight = folder/'model.fp16.ckpt'; cfg_path = folder/'config.yaml'
        if digest(weight) != WEIGHT_SHA or digest(cfg_path) != CONFIG_SHA:
            raise ValueError('Official shape weight/config SHA-256 mismatch')
        signature = {'image_sha256': digest(args.input), 'weight_sha256': WEIGHT_SHA,
                     'config_sha256': CONFIG_SHA, 'source_revision': SOURCE_REVISION,
                     'steps': args.steps, 'seed': args.seed, 'cfg': 5.,
                     'cfg_mode': 'batched' if args.batched_cfg else 'sequential'}
        os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                          HF_HUB_DISABLE_TELEMETRY='1', HY3DGEN_DEBUG='0',
                          USE_SAGEATTN='0', CA_USE_SAGEATTN='0', OMP_NUM_THREADS='6')
        import torch
        import numpy as np
        import yaml
        from PIL import Image
        from accelerate import init_empty_weights
        from safetensors import safe_open
        from safetensors.torch import save_file
        from bounded_attention import bounded_sdpa
        from cpu_volume import CPUVolumeDecoder
        torch.set_num_threads(6); torch.set_num_interop_threads(1)
        if not torch.xpu.is_available():
            raise RuntimeError('Intel XPU unavailable')
        sys.path.insert(0, str(source/'hy3dshape'))
        torch.nn.functional.scaled_dot_product_attention = bounded_sdpa
        from hy3dshape.pipelines import instantiate_from_config, export_to_trimesh
        from hy3dshape.models.autoencoders import MCSurfaceExtractor
        from hy3dshape.models.autoencoders import attention_blocks, attention_processors
        attention_blocks.scaled_dot_product_attention = bounded_sdpa
        attention_processors.scaled_dot_product_attention = bounded_sdpa
        cfg = yaml.safe_load(cfg_path.read_text('utf-8'))
        # mmap avoids simultaneously copying the entire 7.37 GB CPU checkpoint.
        checkpoint = torch.load(weight, map_location='cpu', weights_only=True, mmap=True)
        if set(checkpoint) != {'model', 'vae', 'conditioner'}:
            raise ValueError('Unexpected official checkpoint structure')

        def load(name):
            report('Load '+name+' only')
            with init_empty_weights(include_buffers=False):
                module = instantiate_from_config(cfg[name])
            state = checkpoint[name]
            if name == 'vae':
                del module.encoder, module.pre_kl
                state = {k: v for k, v in state.items() if not k.startswith(('encoder.', 'pre_kl.'))}
            module.load_state_dict(state, strict=True, assign=True)
            module.eval().requires_grad_(False)
            return module.to(device='xpu', dtype=torch.float16)

        latent_file = output/'shape_latents.safetensors'
        resume_file = output/'diffusion_resume.safetensors'
        latent = None
        if args.resume and latent_file.exists():
            with safe_open(str(latent_file), framework='pt', device='cpu') as saved:
                if json.loads(saved.metadata().get('signature', '{}')) != signature:
                    raise ValueError('Latent cache does not match photo/model/steps/seed')
                latent = saved.get_tensor('latents').to('xpu')
            if tuple(latent.shape) != (1, 4096, 64) or not torch.isfinite(latent).all():
                raise ValueError('Invalid shape latents')
        with torch.inference_mode():
            if latent is None:
                completed_steps = 0
                if args.resume and resume_file.exists():
                    with safe_open(str(resume_file), framework='pt', device='cpu') as saved:
                        metadata = saved.metadata()
                        if json.loads(metadata.get('signature', '{}')) != signature:
                            raise ValueError('Diffusion checkpoint does not match photo/model/steps/seed')
                        completed_steps = int(metadata.get('completed_steps', '0'))
                        latent = saved.get_tensor('latents').to('xpu')
                    if not 0 < completed_steps < args.steps:
                        raise ValueError('Invalid diffusion checkpoint step')
                    if tuple(latent.shape) != (1, 4096, 64) or not torch.isfinite(latent).all():
                        raise ValueError('Invalid diffusion checkpoint latents')
                    report('Resume verified diffusion checkpoint', completed_steps=completed_steps)
                conditioner = load('conditioner')
                processor = instantiate_from_config(cfg['image_processor'])
                inputs = processor(Image.open(args.input).convert('RGBA'))
                report('Encode original photo')
                cond = conditioner(**inputs)
                uncond = conditioner.unconditional_embedding(1)
                del conditioner, inputs, processor
                gc.collect(); torch.xpu.empty_cache()
                model = load('model')
                scheduler = instantiate_from_config(cfg['scheduler'])
                scheduler.set_timesteps(sigmas=np.linspace(0, 1, args.steps), device='xpu')
                if latent is None:
                    latent = torch.randn((1, 4096, 64), generator=torch.Generator().manual_seed(args.seed), dtype=torch.float16).to('xpu')
                    latent *= getattr(scheduler, 'init_noise_sigma', 1.)
                else:
                    scheduler.set_begin_index(completed_steps)
                for i, t in enumerate(scheduler.timesteps[completed_steps:], start=completed_steps):
                    report('Infer official 2.1 shape', diffusion_step=i, diffusion_steps=args.steps)
                    ts = t.expand(1).to(latent.dtype)/scheduler.config.num_train_timesteps
                    if args.batched_cfg:
                        def pair(positive_value, negative_value):
                            if isinstance(positive_value, torch.Tensor):
                                return torch.cat((positive_value, negative_value), dim=0)
                            return {key: pair(positive_value[key], negative_value[key])
                                    for key in positive_value}
                        paired = pair(cond, uncond)
                        positive, negative = model(torch.cat((latent, latent), dim=0),
                                                   ts.expand(2), paired).chunk(2)
                        del paired
                    else:
                        positive = model(latent, ts, cond)
                        negative = model(latent, ts, uncond)
                    prediction = negative+5.*(positive-negative)
                    latent = scheduler.step(prediction, t, latent).prev_sample
                    if (i + 1) % 5 == 0 and i + 1 < args.steps:
                        temporary = resume_file.with_suffix('.tmp.safetensors')
                        save_file({'latents': latent.cpu().contiguous()}, str(temporary), metadata={
                            'signature': json.dumps(signature, sort_keys=True),
                            'completed_steps': str(i + 1),
                        })
                        os.replace(temporary, resume_file)
                save_file({'latents': latent.cpu().contiguous()}, str(latent_file),
                          metadata={'signature': json.dumps(signature, sort_keys=True)})
                resume_file.unlink(missing_ok=True)
                del model, cond, uncond, positive, negative, prediction, scheduler
                gc.collect(); torch.xpu.empty_cache()
            vae = load('vae')
            vae.volume_decoder = CPUVolumeDecoder(report)
            vae.surface_extractor = MCSurfaceExtractor()
            report('Decode learned geometry')
            decoded = vae(latent/vae.scale_factor)
            meshes = export_to_trimesh(vae.latents2mesh(decoded, bounds=1.01,
                mc_level=0., num_chunks=1024, octree_resolution=args.resolution))
        if not meshes or meshes[0] is None or not len(meshes[0].faces):
            raise ValueError('No valid shape surface')
        mesh = meshes[0]
        for ext in ['ply', 'glb', 'stl']:
            mesh.export(output/f'generated_raw.{ext}')
        report('Official shape generated; likeness needs visual comparison', state='shape_generated',
               faces=len(mesh.faces), dimensions=mesh.extents.tolist(), watertight=bool(mesh.is_watertight))
        return 0
    except Exception as error:
        import traceback
        traceback.print_exc()
        report('Failed', state='failed', error=str(error))
        return 1


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8'); sys.stderr.reconfigure(encoding='utf-8')
    raise SystemExit(main())
