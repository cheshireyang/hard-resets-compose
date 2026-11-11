import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from lifscan import triton_soft as S
ok = True
for beta in (0.9, 0.95, 0.97, 0.99):
    for mu, sd in ((0.15, 0.4), (-0.1, 0.6), (0.4, 0.3)):
        torch.manual_seed(0)
        T, N = 8192, 333
        I = (mu + sd * torch.randn(T, N, device="cuda")).contiguous(); u0 = (0.3 * torch.randn(N, device="cuda")).contiguous()
        Ur, Sr = S.soft_serial_forward(I, beta, 1.0, u0)
        for C, K in ((16, 16), (32, 32)):
            st = {}
            U, Sp = S.soft_scan_forward(I, beta, 1.0, u0, C=C, K=K, stats=st)
            flip = (Sp != Sr).float().mean().item()
            U2, S2 = S.soft_scan_forward(I, beta, 1.0, u0, C=C, K=K, repair=10**6, stats=st)
            bit = torch.equal(U2, Ur) and torch.equal(S2, Sr); ok &= bit
            print(f"beta={beta} mu={mu} C={C} K={K} flip={flip:.1e} bit={bit} ovf={st['overflow_frac']:.3f} pieces={st['mean_pieces']:.1f} max={st['max_pieces']} repair={st['repair_iters']}", flush=True)
print("SOFT SCAN OK" if ok else "SOFT SCAN FAIL")
