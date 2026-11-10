import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from lifscan import triton_lif as L, fast
torch.manual_seed(0)
T, N, C = 1024, 4096, 128
I = (0.1 + 0.3 * torch.randn(T, N, device="cuda")).contiguous(); u0 = torch.zeros(N, device="cuda")
U, S = L.serial_forward(I, 0.95, 1., 0., u0)
GS = torch.randn_like(I) * 1e-3; GU = torch.zeros_like(I)
Ls = L.serial_backward_tiled(I, u0, U, S, GS, GU, 0.95, 1., 0., BN=32, D=4, num_warps=1)
Lc = L.scan_backward_tiled(I, u0, U, S, GS, GU, 0.95, 1., 0., C=C, BNr=32, D=8, nwr=1)
# float64 reference of the same linear recurrence (same forward)
v = (0.95 * torch.cat([u0[None], U[:-1]]) + I).double()
sg = fast.surrogate_grad(v - 1.0, 4.0); J = 1 - S.double()
a = torch.zeros_like(v); a[:-1] = 0.95 * J[:-1]; b = GS.double() * sg
lam = torch.zeros_like(v); nxt = torch.zeros(N, dtype=torch.float64, device="cuda")
for t in range(T - 1, -1, -1):
    nxt = b[t] + a[t] * nxt; lam[t] = nxt
es = (Ls.double() - lam).abs(); ec = (Lc.double() - lam).abs()
print("rel err vs fp64: serial", (es.norm() / lam.norm()).item(), " scan", (ec.norm() / lam.norm()).item())
print("mean signed err serial", (Ls.double() - lam).mean().item(), "scan", (Lc.double() - lam).mean().item(), "lam mean", lam.mean().item())
pos = torch.arange(T, device="cuda") % C
for p in (0, 1, C // 2, C - 2, C - 1):
    print(f"pos-in-chunk {p:3d}: scan mean abs err {ec[pos == p].mean().item():.3e}  serial {es[pos == p].mean().item():.3e}")
