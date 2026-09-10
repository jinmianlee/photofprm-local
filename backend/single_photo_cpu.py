"""Offline Hunyuan3D-2mini shape experiment, run in the dedicated Conda env.

This exports the actual generated mesh for inspection before print processing.
It never substitutes demo geometry when the model or inference fails.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'models' / 'hunyuan3d-2mini' / 'hunyuan3d-dit-v2-mini'
SOURCE = ROOT / 'tools' / 'hunyuan3d'
WEIGHT_SHA = '3cc66f3bea33e4062b7dbc875ffe1d70c4888914aec3e91b60f94e9bd01b522b'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--steps', type=int, default=30, choices=range(10, 61))
    parser.add_argument('--resolution', type=int, default=255, choices=[127, 191, 255, 383])
    parser.add_argument('--threads', type=int, default=8, choices=range(1, 17))
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--decode-latents', help='Resume mesh extraction from a previously generated local latent file')
    parser.add_argument('--resume', action='store_true', help='Resume the same image and settings from its saved step')
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    metrics = {'state': 'running', 'model': 'Hunyuan3D-2mini', 'device': 'cpu',
               'steps': args.steps, 'resolution': args.resolution,
               'seed': args.seed, 'peak_rss_gib': 0.0, 'stage': 'preflight'}
    report_lock = threading.Lock()

    def report(stage, **fields):
        with report_lock:
            metrics.update(stage=stage, elapsed_seconds=round(time.perf_counter()-started, 2), **fields)
            temporary = output / 'inference.tmp'
            temporary.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), 'utf-8')
            temporary.replace(output / 'inference.json')
        print(json.dumps({'stage': stage, **fields}, ensure_ascii=False), flush=True)

    # No automatic model downloads or remote code at inference time.
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      HF_HUB_DISABLE_TELEMETRY='1', HY3DGEN_DEBUG='0',
                      CA_USE_SAGEATTN='0', USE_SAGEATTN='0',
                      OMP_NUM_THREADS=str(args.threads), OPENBLAS_NUM_THREADS=str(args.threads))
    stop_monitor = threading.Event()
    try:
        weight = MODEL / 'model.fp16.safetensors'
        if not weight.is_file():
            raise RuntimeError('混元 2mini 权重尚未安装完整。')
        report('校验官方权重')
        sha = hashlib.sha256()
        with weight.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8*1024*1024), b''):
                sha.update(chunk)
        if sha.hexdigest() != WEIGHT_SHA:
            raise RuntimeError('模型权重 SHA-256 校验失败，停止加载。')
        import psutil
        import numpy as np
        import torch
        import yaml
        from accelerate import init_empty_weights
        from PIL import Image, ImageOps
        from safetensors import safe_open
        from safetensors.torch import save_file, load_file
        from cpu_volume import CPUVolumeDecoder
        from inference_checkpoint import load_checkpoint, save_checkpoint
        sys.path.insert(0, str(SOURCE))
        from hy3dgen.shapegen.pipelines import Hunyuan3DDiTFlowMatchingPipeline, instantiate_from_config, export_to_trimesh
        from hy3dgen.shapegen.models.autoencoders import MCSurfaceExtractor

        image = ImageOps.exif_transpose(Image.open(args.input)).convert('RGBA')
        alpha = image.getchannel('A')
        if alpha.getextrema()[0] == 255:
            raise RuntimeError('请先提供去背景的透明 PNG，避免背景进入三维形状。')
        if not alpha.getbbox():
            raise RuntimeError('前景为空。')
        signature = {'input_sha256': hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
                     'weight_sha256': WEIGHT_SHA, 'steps': args.steps, 'seed': args.seed,
                     'guidance': 5.0, 'dtype': 'float32'}
        checkpoint = output / 'diffusion_checkpoint.safetensors'
        completed, saved_latent = 0, None
        if args.resume:
            completed, saved_latent = load_checkpoint(checkpoint, signature)
            report('恢复形状推理断点', completed_steps=completed)
            if completed == args.steps:
                args.decode_latents = str(checkpoint)

        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        process = psutil.Process()

        def monitor():
            while not stop_monitor.wait(1):
                rss = process.memory_info().rss / 2**30
                metrics['peak_rss_gib'] = round(max(metrics['peak_rss_gib'], rss), 3)
                if rss > 24:
                    report('进程超过 24GB 内存上限，已停止', state='failed')
                    os._exit(2)

        threading.Thread(target=monitor, daemon=True).start()
        config = yaml.safe_load((MODEL / 'config.yaml').read_text('utf-8'))
        targets = {
            'model': 'hy3dgen.shapegen.models.Hunyuan3DDiT',
            'vae': 'hy3dgen.shapegen.models.ShapeVAE',
            'conditioner': 'hy3dgen.shapegen.models.SingleImageEncoder',
            'scheduler': 'hy3dgen.shapegen.schedulers.FlowMatchEulerDiscreteScheduler',
            'image_processor': 'hy3dgen.shapegen.preprocessors.ImageProcessorV2',
        }
        for name, target in targets.items():
            if config[name]['target'] != target:
                raise RuntimeError(f'Unexpected model config target: {name}')
        # Create empty parameters, then assign official tensors, avoiding a second
        # full set of randomly initialized weights. Nonpersistent buffers stay CPU.
        def load_component(name):
            report('加载 ' + name)
            with init_empty_weights(include_buffers=False):
                module = instantiate_from_config(config[name])
            if name == 'vae':
                # The inference checkpoint omits the training-only encoder.
                del module.encoder
                del module.pre_kl
            with safe_open(weight, framework='pt', device='cpu') as tensors:
                prefix = name + '.'
                state = {key[len(prefix):]: tensors.get_tensor(key) for key in tensors.keys()
                         if key.startswith(prefix)}
            if name == 'vae':
                state = {k:v for k,v in state.items() if not k.startswith(('encoder.', 'pre_kl.'))}
            module.load_state_dict(state, strict=True, assign=True)
            module.eval().requires_grad_(False)
            module.to(device='cpu', dtype=torch.float32)
            del state
            gc.collect()
            return module

        class SingleUsePipeline(Hunyuan3DDiTFlowMatchingPipeline):
            def prepare_latents(self, *a, **kw):
                if saved_latent is not None:
                    return saved_latent.clone()
                return super().prepare_latents(*a, **kw)

            def encode_cond(self, *a, **kw):
                report('读取照片特征')
                result = super().encode_cond(*a, **kw)
                self.conditioner = None
                gc.collect()
                return result

            def _export(self, *a, **kw):
                save_file({'latents': a[0].detach().cpu().contiguous()}, str(output / 'shape_latents.safetensors'))
                self.model = None
                gc.collect()
                report('生成三角网格')
                return super()._export(*a, **kw)

        if args.decode_latents:
            latent_file = Path(args.decode_latents)
            if latent_file.stat().st_size > 1024 * 1024:
                raise ValueError('Unexpected latent file size.')
            latent = load_file(str(latent_file), device='cpu')['latents']
            if tuple(latent.shape) != (1, 512, 64) or not torch.isfinite(latent).all():
                raise ValueError('Unexpected latent tensor.')
            vae = load_component('vae')
            vae.volume_decoder = CPUVolumeDecoder(report)
            vae.surface_extractor = MCSurfaceExtractor()
            with torch.inference_mode():
                decoded = vae(latent.to(torch.float32) / vae.scale_factor)
                result = export_to_trimesh(vae.latents2mesh(decoded, bounds=1.01,
                    mc_level=0.0, num_chunks=2048, octree_resolution=args.resolution))
        else:
            pipeline = SingleUsePipeline(
                model=load_component('model'), vae=load_component('vae'),
                conditioner=load_component('conditioner'),
                scheduler=instantiate_from_config(config['scheduler']),
                image_processor=instantiate_from_config(config['image_processor']),
                device='cpu', dtype=torch.float32)
            pipeline.vae.volume_decoder = CPUVolumeDecoder(report)
            pipeline.vae.surface_extractor = MCSurfaceExtractor()
            def step(index, _timestep, result):
                current = completed + index + 1
                save_checkpoint(checkpoint, result.prev_sample, current, signature)
                report('推断三维形状', diffusion_step=current, diffusion_steps=args.steps)
            result = pipeline(image=image, num_inference_steps=args.steps,
                              sigmas=np.linspace(0, 1, args.steps)[completed:],
                              octree_resolution=args.resolution, num_chunks=2048,
                              generator=torch.Generator(device='cpu').manual_seed(args.seed),
                              callback=step, callback_steps=1, output_type='trimesh')
        if not result or result[0] is None or not len(result[0].faces):
            raise RuntimeError('模型没有生成有效表面。')
        mesh = result[0]
        mesh.export(output / 'generated_raw.ply')
        mesh.export(output / 'generated_raw.glb')
        mesh.export(output / 'generated_raw.stl')
        report('原始形状已生成，等待外观和打印检查', state='shape_generated',
               faces=len(mesh.faces), vertices=len(mesh.vertices),
               watertight=bool(mesh.is_watertight), dimensions=mesh.extents.tolist(),
               print_validated=False, likeness_validated=False)
        return 0
    except Exception as error:
        import traceback
        traceback.print_exc()
        report('生成失败', state='failed', error=str(error))
        return 1
    finally:
        stop_monitor.set()


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    raise SystemExit(main())
