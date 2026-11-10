import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from lifscan import triton_lif as L
ok = True
for beta in (0.5, 0.9, 0.99, 1.0):
    for mu, sd in ((0.15, 0.4), (-0.1, 0.6), (0.4, 0.2)):
        T, N = 8192, 777
        torch.manual_seed(0)
        I = (mu + sd * torch.randn(T, N, device="cuda")).contiguous(); u0 = (0.3 * torch.randn(N, device="cuda")).contiguous()
        Ur, Sr = L.serial_forward(I, beta, 1., -0.1, u0)
        for K, K2 in ((16, 32), (8, 16)):
            st = {}
            U, S = L.scan_forward_v4(I, beta, 1., -0.1, u0, C=64, G=16, K=K, K2=K2, stats=st)
            flip = (S != Sr).float().mean().item()
            U2, S2 = L.scan_forward_v4(I, beta, 1., -0.1, u0, C=64, G=16, K=K, K2=K2, repair=10**6)
            eq = torch.equal(U2, Ur) and torch.equal(S2, Sr); ok &= eq and flip < 1e-4
            print(f"beta={beta} mu={mu} K={K},{K2} flip={flip:.1e} bit={eq} ovf={st['overflow_frac']:.3f} "
                  f"sovf={st['super_overflow_frac']:.3f} pieces={st['mean_pieces']:.1f} super={st['super_mean_pieces']:.1f} (max {st['super_max_pieces']})", flush=True)
print("V4 OK" if ok else "FAIL")
