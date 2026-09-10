"""Check official Intel GPU execution in the isolated Conda environment."""
import json
from pathlib import Path
import time
import torch
import torch.nn.functional as F

output = Path(__file__).resolve().parents[1] / 'data/xpu-runtime-check.json'
report = {'torch': torch.__version__, 'available': torch.xpu.is_available(),
          'inference_validated': False}
try:
    if not report['available']:
        raise RuntimeError('The official XPU runtime cannot access the Intel GPU.')
    report['device'] = torch.xpu.get_device_name(0)
    report['device_count'] = torch.xpu.device_count()
    generator = torch.Generator().manual_seed(42)
    q, k, v = [torch.randn(1, 16, 512, 64, generator=generator) for _ in range(3)]
    # Compare a real attention-sized operation against the CPU reference.
    reference = F.scaled_dot_product_attention(q, k, v)
    q, k, v = [value.to(device='xpu', dtype=torch.float16) for value in (q, k, v)]
    started = time.perf_counter()
    result = F.scaled_dot_product_attention(q, k, v).cpu().float()
    torch.xpu.synchronize()
    report['attention_seconds'] = round(time.perf_counter()-started, 3)
    error = (result-reference).abs()
    report['attention_max_error'] = error.max().item()
    report['attention_mean_error'] = error.mean().item()
    torch.testing.assert_close(result, reference, rtol=.03, atol=.002)
    a = torch.randn(256, 256, generator=generator)
    b = torch.randn(256, 256, generator=generator)
    actual = (a.to('xpu') @ b.to('xpu')).cpu()
    torch.testing.assert_close(actual, a @ b, rtol=.002, atol=.002)
    report['operations_validated'] = True
except Exception as error:
    report['operations_validated'] = False
    report['error'] = str(error)
finally:
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), 'utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
raise SystemExit(0 if report['operations_validated'] else 1)
