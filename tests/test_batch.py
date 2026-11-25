"""GPU test: per-column-parameter serial kernels (lifscan.triton_batch) == scalar-parameter kernels column by column
(bit-exact forward and adjoints), and per-column parameter gradients == the scalar layers' gradients."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from lifscan import triton_lif as L, triton_alif as A, triton_batch as B
from lifscan.fit import lif_param

dev = "cuda"; ok = True
torch.manual_seed(0)
T, N = 8192, 24
I = (0.12 + 0.4 * torch.randn(T, N, device=dev)).contiguous(); u0 = torch.zeros(N, device=dev)
beta = 0.85 + 0.14 * torch.rand(N, device=dev); ur = 0.3 * torch.rand(N, device=dev) - 0.15
GS = torch.randn(T, N, device=dev); GU = torch.zeros_like(GS)
U, S = B.lif_vec_forward(I, beta, ur, u0)
lam = B.lif_vec_backward(I, u0, U, S, GS, beta, ur)
for n in range(N):
    In = I[:, n:n + 1].contiguous(); b, r = float(beta[n]), float(ur[n])
    Un, Sn = L.serial_forward(In, b, 1.0, r, u0[:1])
    Ln = L.serial_backward_tiled_pf(In, u0[:1], Un, Sn, GS[:, n:n + 1].contiguous(), GU[:, :1].contiguous(), b, 1.0, r)
    ok &= torch.equal(Un[:, 0], U[:, n]) and torch.equal(Sn[:, 0], S[:, n]) and torch.equal(Ln[:, 0], lam[:, n])
print("LIF fwd/bwd bit-exact per column:", ok)
# parameter gradients vs the scalar autograd layer
bp = beta.clone().requires_grad_(True); urp = ur.clone().requires_grad_(True); Ip = I.clone().requires_grad_(True)
(B.lif_vec(Ip, u0, bp, urp) * GS).sum().backward()
gmax = 0.0
for n in range(N):
    b1 = beta[n].clone().requires_grad_(True); r1 = ur[n].clone().requires_grad_(True)
    S1 = lif_param(I[:, n:n + 1].contiguous(), u0[:1], b1, torch.tensor(1.0, device=dev), r1, mode="serial")
    (S1 * GS[:, n:n + 1]).sum().backward()
    gmax = max(gmax, abs(b1.grad - bp.grad[n]).item() / max(1e-6, abs(b1.grad).item()), abs(r1.grad - urp.grad[n]).item() / max(1e-6, abs(r1.grad).item()))
ok &= gmax < 1e-4
print(f"LIF parameter gradients: max rel diff {gmax:.1e}")
# ALIF
rho = 0.95 + 0.04 * torch.rand(N, device=dev); q = 0.05 + 0.3 * torch.rand(N, device=dev)
Ua, Wa, Sa = B.alif_vec_forward(I, beta, rho, q, ur, u0, u0)
LVa, LWa = B.alif_vec_backward(I, Ua, Wa, Sa, GS, beta, rho, q, u0, u0)
oka = True
for n in range(N):
    In = I[:, n:n + 1].contiguous(); b, rr, qq, r = float(beta[n]), float(rho[n]), float(q[n]), float(ur[n])
    Un, Wn, Sn = A.alif_serial_forward(In, b, rr, qq, 1.0, r, u0[:1], u0[:1])
    LV, LW = A.alif_backward(In, Un, Wn, Sn, GS[:, n:n + 1].contiguous(), b, rr, qq, 1.0, u0[:1], u0[:1], mode="serial_tuned")
    oka &= torch.equal(Un[:, 0], Ua[:, n]) and torch.equal(Wn[:, 0], Wa[:, n]) and torch.equal(Sn[:, 0], Sa[:, n])
    oka &= torch.allclose(LV[:, 0], LVa[:, n], rtol=1e-5, atol=1e-6) and torch.allclose(LW[:, 0], LWa[:, n], rtol=1e-5, atol=1e-6)
print("ALIF fwd bit-exact, adjoints match per column:", oka)
ok &= oka
print("BATCH OK" if ok else "BATCH FAIL")
sys.exit(0 if ok else 1)
