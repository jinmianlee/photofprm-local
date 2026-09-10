"""Compare bounded attention with the complete attention operation."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))
import torch
from bounded_attention import bounded_sdpa, NATIVE_SDPA

torch.set_num_threads(4)
generator = torch.Generator().manual_seed(7)
for device, dtype in [('cpu', torch.float32), ('xpu', torch.float16)]:
    q = torch.randn(1, 4, 1537, 64, generator=generator).to(device=device, dtype=dtype)
    k = torch.randn(1, 4, 1703, 64, generator=generator).to(device=device, dtype=dtype)
    v = torch.randn(1, 4, 1703, 64, generator=generator).to(device=device, dtype=dtype)
    expected = NATIVE_SDPA(q, k, v)
    actual = bounded_sdpa(q, k, v)
    torch.testing.assert_close(actual, expected, rtol=.005, atol=.001)
    print(device, 'maximum difference', (actual-expected).abs().max().item(), flush=True)
print('BOUNDED_ATTENTION_OK', flush=True)
