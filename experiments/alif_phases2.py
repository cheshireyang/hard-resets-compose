import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from lifscan import triton_alif as A
name, beta, rho, q, mfree, N, T = sys.argv[1], *map(float, sys.argv[2:6]), int(sys.argv[6]), int(sys.argv[7])
torch.manual_seed(0)
I = ((1 - beta) * mfree + 0.5 * math.sqrt(1 - beta * beta) * torch.randn(T, N, device="cuda")).contiguous()
u0 = torch.zeros(N, device="cuda"); w0 = torch.zeros(N, device="cuda")
_, Wr, _ = A.alif_serial_forward(I, beta, rho, q, 1.0, 0.0, u0, w0)
_, Wp, _ = A.alif_serial_forward((1.005 * I).contiguous(), beta, rho, q, 1.0, 0.0, u0, w0)
for C in (64, 128, 256):
    for tag, src in (("oracle", Wr), ("prev", Wp)):
        Wpred = torch.cat([w0[None], src[C - 1:-1:C]], 0).contiguous()
        for dwq in (0.1, 0.3, 1.0):
            K2 = 32 if dwq < 1 else 64
            for rep in range(3):
                ph = {}; st = {}
                A.alif_scan_forward(I, beta, rho, q, 1.0, 0.0, u0, w0, Wpred, dwq * q, C=C, K1=32, K2=K2, BN=4, nw=1, BNc=1, D=4, phases=ph, stats=st)
            print(f"C={C:3d} {tag:6s} dw={dwq}q: " + "  ".join(f"{k} {v:.2f}" for k, v in ph.items()) + f"  total {sum(ph.values()):.2f} ms | miss {100*st['miss_frac']:.1f}% ovf {100*st['overflow_frac']:.1f}% pieces {st['mean_pieces']:.1f}", flush=True)
