import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from lifscan import triton_affine as A
from lifscan import fast
ok = True
for gamma, delta in ((0.0, 0.0), (0.0, -0.2), (0.3, -0.1), (0.7, -0.5), (1.0, -1.0), (0.5, 0.2)):
    for beta in (0.9, 0.97):
        torch.manual_seed(0)
        T, N = 8192, 200
        I = (0.15 + 0.4 * torch.randn(T, N, device="cuda")).contiguous(); u0 = (0.2 * torch.randn(N, device="cuda")).contiguous()
        Ur, Sr = A.aff_serial_forward(I, beta, 1.0, gamma, delta, u0)
        st = {}
        U, S = A.aff_scan_forward(I, beta, 1.0, gamma, delta, u0, C=128, K=32, stats=st)
        flip = (S != Sr).float().mean().item()
        U2, S2 = A.aff_scan_forward(I, beta, 1.0, gamma, delta, u0, C=128, K=32, repair=10**6, stats=st)
        bit = torch.equal(U2, Ur) and torch.equal(S2, Sr); ok &= bit
        print(f"gamma={gamma} delta={delta} beta={beta}: flip={flip:.1e} bit={bit} ovf={st['overflow_frac']:.3f} pieces={st['mean_pieces']:.1f} repair={st['repair_iters']}", flush=True)
# gradients vs float64 autograd
T, N = 4096, 8
x = torch.randn(T, N, device="cuda"); W = torch.randn(T, N, device="cuda")
for mode in ("serial", "scan"):
    ps = [torch.tensor(v, device="cuda", requires_grad=True) for v in (0.95, 0.4, -0.3, 0.3)]   # beta, gamma, delta, gain
    S = A.lif_affine(ps[3] * x + 0.08, torch.zeros(N, device="cuda"), ps[0], ps[1], ps[2], mode=mode)
    (W * S).sum().backward()
    q = [torch.tensor(v, dtype=torch.float64, device="cuda", requires_grad=True) for v in (0.95, 0.4, -0.3, 0.3)]
    Id = q[3] * x.double() + 0.08; u = torch.zeros(N, dtype=torch.float64, device="cuda"); ss = []
    for t in range(T):
        v = q[0] * u + Id[t]; s = fast.SpikeFn.apply(v - 1.0, 4.0); sd = s.detach()
        u = (1 - sd) * v + sd * (q[1] * v + q[2]); ss.append(s)
    ss = torch.stack(ss); (W.double() * ss).sum().backward()
    same = torch.equal(S.detach(), ss.float().detach())
    rel = [abs(a.grad.item() - b.grad.item()) / (abs(b.grad.item()) + 1e-9) for a, b in zip(ps, q)]
    print(mode, "spikes equal fp64:", same, "rel grad err (beta,gamma,delta,gain):", [f"{r:.1e}" for r in rel])
    ok &= max(rel) < 1e-3
print("AFFINE OK" if ok else "AFFINE FAIL")
