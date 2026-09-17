"""Offline XPU ablation of official Standard mini; not a quality guarantee.

Verifies original weights/config before changing inference token/image sizes.
Serial CFG avoids doubling peak device memory. No learned tensors are changed.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'))
from runtime_config import config
from file_io import write_json

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--tokens',type=int,choices=[512,1024,3072],default=512)
    p.add_argument('--conditioning-size',type=int,choices=[518,1022],default=518)
    p.add_argument('--steps',type=int,choices=[30,50],default=30)
    p.add_argument('--resolution',type=int,choices=[255,383],default=255)
    args=p.parse_args()
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    c=config();model=c['model_dir'].parent/'hunyuan3d-dit-v2-mini'
    start=time.monotonic();metrics={'model':'Hunyuan3D-2mini-Standard','state':'running','steps':args.steps,'tokens':args.tokens,'resolution':args.resolution,'seed':12345,'conditioning_pixels':args.conditioning_size,'cfg':5.,'experimental':args.tokens!=512 or args.conditioning_size!=518}
    def report(stage,**fields):
        metrics.update(stage=stage,elapsed_seconds=round(time.monotonic()-start,2),**fields)
        write_json(out/'inference.json',metrics);print(json.dumps({'stage':stage,**fields},ensure_ascii=False),flush=True)
    try:
        report('Verify official Standard mini files')
        for name,sha in [('model.fp16.safetensors','3cc66f3bea33e4062b7dbc875ffe1d70c4888914aec3e91b60f94e9bd01b522b'),('config.yaml','cabcba7f6115752c8fe5b370e12bf714936f70377a8a80f151872f76c2d64609')]:
            with (model/name).open('rb') as f:
                if hashlib.file_digest(f,'sha256').hexdigest()!=sha:raise ValueError('Official file checksum failed: '+name)
        os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',CA_USE_SAGEATTN='0',USE_SAGEATTN='0',HY3DGEN_DEBUG='0',OMP_NUM_THREADS='6')
        import torch
        import numpy as np
        import yaml
        from PIL import Image
        from safetensors.torch import save_file
        sys.path.insert(0,str(c['source_dir']))
        from hy3dgen.shapegen.pipelines import export_to_trimesh
        from hy3dgen.shapegen.models.autoencoders import MCSurfaceExtractor
        from model_loading import load_pipeline
        from bounded_attention import install
        from cpu_volume import CPUVolumeDecoder
        install();torch.set_num_threads(6);torch.set_num_interop_threads(1)
        if not torch.xpu.is_available():raise RuntimeError('XPU unavailable')
        cfg=yaml.safe_load((model/'config.yaml').read_text('utf-8'))
        cfg['vae']['params']['num_latents']=args.tokens
        cfg['conditioner']['params']['main_image_encoder']['kwargs']['image_size']=args.conditioning_size
        if args.conditioning_size==1022:cfg['image_processor']['params']['size']=1022
        report('Load Standard mini, enlarged conditioning')
        pipe=load_pipeline(cfg,str(model/'model.fp16.safetensors'),'xpu',torch.float16)
        with torch.inference_mode():
            report('Encode image')
            image_inputs=pipe.prepare_image(Image.open(args.input).convert('RGBA'));im=image_inputs.pop('image')
            full=pipe.encode_cond(im,image_inputs,do_classifier_free_guidance=True,dual_guidance=False)
            cond={k:v[:1] for k,v in full.items()};uncond={k:v[1:] for k,v in full.items()}
            del full;pipe.conditioner=None;gc.collect();torch.xpu.empty_cache()
            pipe.scheduler.set_timesteps(sigmas=np.linspace(0,1,args.steps),device='xpu')
            latent=pipe.prepare_latents(1,torch.float16,'xpu',torch.Generator().manual_seed(12345))
            for i,t in enumerate(pipe.scheduler.timesteps):
                report('Standard shape inference',diffusion_step=i,diffusion_steps=args.steps)
                timestep=t.expand(1).to(latent.dtype)/pipe.scheduler.config.num_train_timesteps
                positive=pipe.model(latent,timestep,cond,guidance=None)
                negative=pipe.model(latent,timestep,uncond,guidance=None)
                pred=negative+5.*(positive-negative)
                latent=pipe.scheduler.step(pred,t,latent).prev_sample
            save_file({'latents':latent.cpu().contiguous()},str(out/'shape_latents.safetensors'))
            vae=pipe.vae
            del pipe,cond,uncond,positive,negative,pred;gc.collect();torch.xpu.empty_cache()
            vae.volume_decoder=CPUVolumeDecoder(report);vae.surface_extractor=MCSurfaceExtractor()
            decoded=vae(latent/vae.scale_factor)
            mesh=export_to_trimesh(vae.latents2mesh(decoded,bounds=1.01,mc_level=0.,num_chunks=2048,octree_resolution=args.resolution))[0]
        if mesh is None or not len(mesh.faces):raise ValueError('Empty Standard mini surface')
        for ext in ['ply','glb','stl']:mesh.export(out/f'generated_raw.{ext}')
        report('Standard shape generated',state='shape_generated',faces=len(mesh.faces),watertight=bool(mesh.is_watertight),likeness_validated=False)
    except Exception as error:
        import traceback;traceback.print_exc();report('Failed',state='failed',error=str(error));return 1
    return 0

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');sys.stderr.reconfigure(encoding='utf-8')
    raise SystemExit(main())
