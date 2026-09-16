"""Verify installed files and real XPU operations; does not claim image quality."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.runtime_config import config, fingerprint, WEIGHT_SHA, CONFIG_SHA, CODE_REVISION
from backend.file_io import write_json


def main():
    report = {'ready': False, 'checked_at': time.time(), 'inference_validated': False,
              'checks': [], 'quality_note': 'Runtime preflight only; inspect each generated model before printing.'}
    try:
        c = config()
        if Path(sys.prefix).resolve() != c['ai_env']:
            raise RuntimeError('Run this check using the configured AI Conda environment.')
        for name, expected in [('model.fp16.safetensors', WEIGHT_SHA), ('config.yaml', CONFIG_SHA)]:
            with (c['model_dir']/name).open('rb') as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
                    raise RuntimeError(f'Official file hash mismatch: {name}')
            report['checks'].append(name + ': sha256 verified')
        from backend.foreground import verify
        verify(c['rembg_dir']/'u2net.onnx')
        head = subprocess.check_output(['git', '-C', str(c['source_dir']), 'rev-parse', 'HEAD'], text=True).strip()
        if head != CODE_REVISION:
            raise RuntimeError('Hunyuan source revision does not match the supported version.')
        dirty = subprocess.check_output(['git', '-C', str(c['source_dir']), 'status', '--porcelain', '--untracked-files=no'], text=True)
        if dirty.strip():
            raise RuntimeError('Official Hunyuan source has local changes; review before using this runtime.')
        os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1', U2NET_HOME=str(c['rembg_dir']))
        import torch
        import rembg
        import safetensors
        import skimage
        sys.path.insert(0, str(c['source_dir']))
        from hy3dgen.shapegen.pipelines import Hunyuan3DDiTFlowMatchingPipeline
        if not torch.xpu.is_available():
            raise RuntimeError('Intel XPU unavailable. This deployment currently requires a supported Intel GPU.')
        q = torch.randn(1, 4, 64, 32, generator=torch.Generator().manual_seed(5))
        reference = torch.nn.functional.scaled_dot_product_attention(q, q, q)
        x = q.to(device='xpu', dtype=torch.float16)
        result = torch.nn.functional.scaled_dot_product_attention(x, x, x).cpu().float()
        torch.testing.assert_close(result, reference, rtol=.03, atol=.004)
        report.update(ready=True, torch=torch.__version__, device=torch.xpu.get_device_name(0),
                      fingerprint=fingerprint(c))
        report['checks'].extend(['foreground hash verified', 'official source revision verified', 'pipeline imports OK', 'XPU attention agrees with CPU'])
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
    write_json(ROOT/'data/engine-check.json', report)
    print('Engine check: ' + ('PASS' if report['ready'] else report['error']), flush=True)
    return 0 if report['ready'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
