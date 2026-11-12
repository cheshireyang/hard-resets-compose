"""ALIF gradients: serial and scan backward vs float64 autograd BPTT (same surrogate, detached reset)."""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from lifscan import triton_alif as A
from lifscan import fast
dev = "cuda"; ok = True
for beta, rho, q in ((0.95, 0.995, 0.9), (0.99, 0.96, 0.2)):
    torch.manual_seed(0)
    T, N = 4096, 8
    x = (0.06 + 0.15 * torch.randn(T, N, device=dev)).contiguous(); G = torch.randn(T, N, device=dev)
    vals = (beta, rho, q, 1.0)
    # reference: float64 autograd through the plain recurrence
    qd = [torch.tensor(v, dtype=torch.float64, device=dev, requires_grad=True) for v in vals]
    xd = x.double().requires_grad_(True)
    u = torch.zeros(N, dtype=torch.float64, device=dev); w = torch.zeros(N, dtype=torch.float64, device=dev); ss = []
    for t in range(T):
        v = qd[0] * u + xd[t]
        s = fast.SpikeFn.apply(v - qd[3] - w, 4.0); sd = s.detach()
        u = (1 - sd) * v + sd * 0.0
        w = qd[1] * w + qd[2] * s
        ss.append(s)
    Sd = torch.stack(ss); (G.double() * Sd).sum().backward()
    ref = [p.grad.item() for p in qd]
    for mode in ("serial", "scan"):
        ps = [torch.tensor(v, device=dev, requires_grad=True) for v in vals]
        xi = x.clone().requires_grad_(True)
        S, U, W = A.alif(xi, torch.zeros(N, device=dev), torch.zeros(N, device=dev), ps[0], ps[1], ps[2], ps[3], fwd="serial", bwd=mode)
        (G * S).sum().backward()
        same = torch.equal(S.detach(), Sd.float().detach())
        rel = [abs(a.grad.item() - b) / (abs(b) + 1e-9) for a, b in zip(ps, ref)]
        gi = (xi.grad.double() - xd.grad).abs().max().item() / xd.grad.abs().max().item()
        print(f"beta={beta} rho={rho} q={q} {mode:6s}: spikes equal fp64 {same}; rel grad err beta,rho,q,theta "
              + " ".join(f"{r:.1e}" for r in rel) + f"; dI max rel {gi:.1e}", flush=True)
        ok &= same and max(rel) < 1e-3 and gi < 1e-3
print("ALIF GRAD OK" if ok else "ALIF GRAD FAIL")
