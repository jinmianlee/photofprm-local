"""Offline 2mini Turbo worker using the verified Intel XPU runtime."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from file_io import write_json
from shape_cache import generation_signature, record_shape
from runtime_config import config as runtime_config

ROOT = Path(__file__).resolve().parents[1]
MODEL = runtime_config()['model_dir']
WEIGHT_SHA = 'bdbcef30dd0149a281e17d5b5b1fdad1122c904e098a42f3100e04e03c247bc4'
CONFIG_SHA = 'be28205844da01bd5d3c5ba5160f5886fb9765d542483f9d12bda8f17324db5e'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--efficient-loading', action='store_true', help='Assign official tensors without disposable parameter initialization')
    parser.add_argument('--resolution', type=int, choices=[127, 255, 383], default=255)
    parser.add_argument('--steps', type=int, choices=[5, 10, 15], default=5)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--detail-roi', type=float, nargs=4, help='Experimental ROI in the prepared square image; not enabled in the web app')
    parser.add_argument('--detail-strength', type=float, default=.35)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    metrics = {'state': 'running', 'model': 'Hunyuan3D-2mini-Turbo', 'device': 'Intel Arc / XPU',
               'steps': args.steps, 'resolution': args.resolution, 'seed': args.seed, 'peak_rss_sampled_gib': 0.0,
               'memory_sampling': 'at_progress_updates_not_continuous',
               'weight_loading': 'direct_assignment' if args.efficient_loading else 'official_standard'}
    process = None
    previous_stage = None
    phase_started = time.monotonic()
    phase_durations = {}

    def report(stage, **fields):
        nonlocal previous_stage, phase_started
        now = time.monotonic()
        if stage != previous_stage:
            if previous_stage:
                phase_durations[previous_stage] = round(phase_durations.get(previous_stage, 0)+now-phase_started, 2)
            previous_stage, phase_started = stage, now
        metrics['phase_seconds'] = {**phase_durations, stage: round(now-phase_started, 2)}
        metrics.update(stage=stage, elapsed_seconds=round(time.monotonic()-started, 2), **fields)
        if process:
            metrics['peak_rss_sampled_gib'] = max(metrics['peak_rss_sampled_gib'], round(process.memory_info().rss / 2**30, 3))
        try:
            write_json(output / 'inference.json', metrics)
        except OSError as error:
            print(f'Progress update delayed; computation continues: {error}', flush=True)
        print(json.dumps({'stage': stage, **fields}, ensure_ascii=False), flush=True)

    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1',
                      HY3DGEN_DEBUG='0', CA_USE_SAGEATTN='0', USE_SAGEATTN='0', OMP_NUM_THREADS='6')
    try:
        report('校验官方 Turbo 权重')
        weight = MODEL / 'model.fp16.safetensors'
        with weight.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != WEIGHT_SHA:
                raise ValueError('Turbo 权重 SHA-256 校验失败，停止加载。')
        config_path = MODEL / 'config.yaml'
        if hashlib.sha256(config_path.read_bytes()).hexdigest() != CONFIG_SHA:
            raise ValueError('Turbo 配置校验失败，停止加载。')
        report('初始化本地推理环境')
        import psutil
        import torch
        import yaml
        from PIL import Image
        from safetensors import safe_open
        from safetensors.torch import save_file
        from inference_checkpoint import load_checkpoint, save_checkpoint
        from cpu_volume import CPUVolumeDecoder
        sys.path.insert(0, str(runtime_config()['source_dir']))
        from hy3dgen.shapegen.pipelines import Hunyuan3DDiTFlowMatchingPipeline, instantiate_from_config, export_to_trimesh
        from hy3dgen.shapegen.models.autoencoders import MCSurfaceExtractor
        from bounded_attention import install
        install()
        torch.set_num_threads(6)
        torch.set_num_interop_threads(1)
        if not torch.xpu.is_available():
            raise RuntimeError('Intel GPU 当前不可用，请查看 XPU 环境检查。')
        process = psutil.Process()
        signature = generation_signature(args.input, args.steps, args.seed)
        if args.detail_roi:
            signature['conditioning_adapter'] = {'method': 'multiscale_dino_roi_v1',
                'roi': args.detail_roi, 'strength': args.detail_strength}
        checkpoint = output / 'diffusion_checkpoint.safetensors'
        completed, latent = load_checkpoint(checkpoint, signature) if args.resume else (0, None)
        dtype, device = torch.float16, 'xpu'
        if completed < args.steps:
            report('加载混元 2mini Turbo')
            if args.efficient_loading:
                from model_loading import load_pipeline
                pipeline = load_pipeline(yaml.safe_load(config_path.read_text('utf-8')), str(weight), device, dtype)
            else:
                pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_single_file(
                    str(weight), str(config_path), device=device, dtype=dtype, use_safetensors=True)
            with torch.inference_mode():
                report('读取照片特征')
                inputs = pipeline.prepare_image(Image.open(args.input).convert('RGBA'))
                photo = inputs.pop('image')
                cond = pipeline.encode_cond(image=photo, additional_cond_inputs=inputs,
                                            do_classifier_free_guidance=False, dual_guidance=False)
                if args.detail_roi:
                    report('实验：编码重点区域的细节')
                    from detail_conditioning import enrich_conditioning
                    cond, detail_report = enrich_conditioning(pipeline, photo, cond, args.detail_roi, args.detail_strength)
                    write_json(output / 'detail_conditioning.json', detail_report)
                pipeline.conditioner = None
                gc.collect()
                torch.xpu.empty_cache()
                # Keep the full original schedule when resuming. Passing fewer
                # sigmas to this scheduler would create a different schedule.
                pipeline.scheduler.set_timesteps(num_inference_steps=args.steps, device=device)
                pipeline.scheduler.set_begin_index(completed)
                if latent is None:
                    latent = pipeline.prepare_latents(1, dtype, device, torch.Generator().manual_seed(args.seed))
                else:
                    latent = latent.to(device=device, dtype=dtype)
                guidance = torch.tensor([5.0], device=device, dtype=dtype)
                report('推断三维形状', diffusion_step=completed, diffusion_steps=args.steps)
                for index in range(completed, args.steps):
                    t = pipeline.scheduler.timesteps[index]
                    timestep = t.expand(1).to(dtype) / pipeline.scheduler.config.num_train_timesteps
                    prediction = pipeline.model(latent, timestep, cond, guidance=guidance)
                    latent = pipeline.scheduler.step(prediction, t, latent).prev_sample
                    save_checkpoint(checkpoint, latent, index+1, signature)
                    report('推断三维形状', diffusion_step=index+1, diffusion_steps=args.steps)
                vae = pipeline.vae
                del prediction, cond, pipeline
                gc.collect()
                torch.xpu.empty_cache()
        else:
            report('恢复网格提取断点')
            config = yaml.safe_load(config_path.read_text('utf-8'))
            if args.efficient_loading:
                from model_loading import load_component
                vae = load_component('vae', config, str(weight), device, dtype)
            else:
                vae = instantiate_from_config(config['vae'])
                del vae.encoder
                del vae.pre_kl
                with safe_open(weight, framework='pt', device='cpu') as handle:
                    state = {k[4:]: handle.get_tensor(k) for k in handle.keys()
                             if k.startswith('vae.') and not k.startswith(('vae.encoder.', 'vae.pre_kl.'))}
                vae.load_state_dict(state, strict=True)
                del state
                vae.to(device=device, dtype=dtype)
            latent = latent.to(device=device, dtype=dtype)
        vae.eval().requires_grad_(False)
        save_file({'latents': latent.detach().cpu().contiguous()}, str(output / 'shape_latents.safetensors'))
        vae.volume_decoder = CPUVolumeDecoder(report)
        vae.surface_extractor = MCSurfaceExtractor()
        with torch.inference_mode():
            report('生成三角网格')
            decoded = vae(latent / vae.scale_factor)
            meshes = export_to_trimesh(vae.latents2mesh(decoded, bounds=1.01, mc_level=0.0,
                                        num_chunks=2048, octree_resolution=args.resolution))
        if not meshes or meshes[0] is None or not len(meshes[0].faces):
            raise ValueError('模型没有生成有效表面。')
        mesh = meshes[0]
        for extension in ('ply', 'glb', 'stl'):
            mesh.export(output / f'generated_raw.{extension}')
        record_shape(output, signature, args.resolution, mesh)
        report('原始形状已生成，等待打印处理', state='shape_generated', faces=len(mesh.faces),
               vertices=len(mesh.vertices), dimensions=mesh.extents.tolist(), watertight=bool(mesh.is_watertight),
               print_validated=False, likeness_validated=False)
        return 0
    except Exception as error:
        import traceback
        traceback.print_exc()
        report('生成失败', state='failed', error=str(error))
        return 1


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    raise SystemExit(main())
