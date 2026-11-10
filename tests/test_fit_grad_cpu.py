"""CPU: parameter gradients of the scan-based layer vs autograd through the serial loop (float64)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from lifscan.fit import lif_param_cpu
from lifscan import fast
torch.manual_seed(0)
T, N = 2048, 16
x = torch.randn(T, N, dtype=torch.float64)
W = torch.randn(T, N, dtype=torch.float64)
def params():
    return [torch.tensor(v, dtype=torch.float64, requires_grad=True) for v in (0.93, 1.0, -0.1, 0.5)]
b, th, ur, g = params()
S = lif_param_cpu(g * x + 0.08, torch.zeros(N, dtype=torch.float64), b, th, ur, C=64)
(W * S).sum().backward(); ours = [p.grad.item() for p in (b, th, ur, g)]
b2, th2, ur2, g2 = params()
us, ss = fast.serial_lif(g2 * x + 0.08, b2, th2, ur2, torch.zeros(N, dtype=torch.float64), alpha=4.0, detach_reset=True)
(W * ss).sum().backward(); ref = [p.grad.item() for p in (b2, th2, ur2, g2)]
rel = [abs(a - r) / abs(r) for a, r in zip(ours, ref)]
print("spikes equal:", torch.equal(S.detach(), ss.detach()))
for n, a, r, e in zip(("beta", "theta", "u_r", "gain"), ours, ref, rel):
    print(f"  d/d{n:5s} ours {a: .10e}  autograd {r: .10e}  rel {e:.1e}")
assert max(rel) < 1e-9, rel
print("FIT GRAD (CPU) OK")
