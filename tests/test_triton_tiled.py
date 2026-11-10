import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from lifscan import triton_lif as L
ok = True
for N, T in ((100, 4096), (1000, 2048), (5000, 1024)):
    torch.manual_seed(1)
    I = (0.15 + 0.4 * torch.randn(T, N, device="cuda")).contiguous(); u0 = (0.2*torch.randn(N, device="cuda")).contiguous()
    Ur, Sr = L.serial_forward(I, 0.9, 1., -0.1, u0)
    GS = torch.randn_like(I); GU = torch.randn_like(I)
    Lr = L.serial_backward(I, u0, Ur, Sr, GS, GU, 0.9, 1., -0.1, detach=False)
    for BN, D, nw in ((32, 4, 1), (64, 8, 2), (128, 2, 4)):
        U, S = L.serial_forward_tiled(I, 0.9, 1., -0.1, u0, BN=BN, D=D, num_warps=nw)
        Lt = L.serial_backward_tiled(I, u0, U, S, GS, GU, 0.9, 1., -0.1, detach=False, BN=BN, D=D, num_warps=nw)
        U2, S2 = L.scan_forward_tiled(I, 0.9, 1., -0.1, u0, C=64, K=32, BNr=BN, D=D, nwr=nw, repair=10**6)
        L2 = L.scan_backward_tiled(I, u0, Ur, Sr, GS, GU, 0.9, 1., -0.1, detach=False, C=64, BNr=BN, D=D, nwr=nw)
        e = [torch.equal(U, Ur) and torch.equal(S, Sr), ((Lt - Lr).norm() / Lr.norm()).item(),
             torch.equal(U2, Ur) and torch.equal(S2, Sr), ((L2 - Lr).norm() / Lr.norm()).item()]
        print(N, T, BN, D, nw, e)
        ok &= e[0] and e[2] and e[1] < 1e-6 and e[3] < 1e-5
print("TILED OK" if ok else "FAIL")
