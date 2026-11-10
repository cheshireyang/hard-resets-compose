"""GPU: parameter gradients of lif_param vs float64 autograd through the serial recurrence."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from lifscan.fit import lif_param
from lifscan import fast
torch.manual_seed(0)
T, N = 4096, 64
x = torch.randn(T, N, device="cuda")
ok = True
for mode in ("serial", "scan"):
    beta = torch.tensor(0.93, device="cuda", requires_grad=True)
    theta = torch.tensor(1.0, device="cuda", requires_grad=True)
    ur = torch.tensor(-0.1, device="cuda", requires_grad=True)
    gain = torch.tensor(0.5, device="cuda", requires_grad=True)
    I = gain * x + 0.08
    u0 = torch.zeros(N, device="cuda")
    W = torch.randn(T, N, device="cuda")
    S = lif_param(I, u0, beta, theta, ur, mode=mode)
    (W * S).sum().backward()
    g = [p.grad.item() for p in (beta, theta, ur, gain)]
    # float64 reference with the same spikes (float64 trajectory may flip spikes; compare where it does not)
    b64 = torch.tensor(0.93, dtype=torch.float64, device="cuda", requires_grad=True)
    t64 = torch.tensor(1.0, dtype=torch.float64, device="cuda", requires_grad=True)
    r64 = torch.tensor(-0.1, dtype=torch.float64, device="cuda", requires_grad=True)
    g64 = torch.tensor(0.5, dtype=torch.float64, device="cuda", requires_grad=True)
    I64 = g64 * x.double() + 0.08
    us, ss = fast.serial_lif(I64, b64, t64, r64, u0.double(), alpha=4.0, detach_reset=True)
    (W.double() * ss).sum().backward()
    ref = [p.grad.item() for p in (b64, t64, r64, g64)]
    same = torch.equal(ss.float(), S.detach())
    rel = [abs(a - b) / (abs(b) + 1e-12) for a, b in zip(g, ref)]
    print(mode, "spikes equal fp64:", same, "grads", [f"{a:.6g}" for a in g], "ref", [f"{b:.6g}" for b in ref], "rel", [f"{r:.1e}" for r in rel])
    ok &= same and max(rel) < 1e-3
print("FIT GRAD OK" if ok else "FIT GRAD CHECK FAILED")
