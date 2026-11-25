import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from lifscan import triton_alif as A
name, beta, rho, q, mfree, N, T = sys.argv[1], *map(float, sys.argv[2:6]), int(sys.argv[6]), int(sys.argv[7])
torch.manual_seed(0)
I = ((1 - beta) * mfree + 0.5 * math.sqrt(1 - beta * beta) * torch.randn(T, N, device="cuda")).contiguous()
u0 = torch.zeros(N, device="cuda"); w0 = torch.zeros(N, device="cuda")
_, Wp, _ = A.alif_serial_forward((1.005 * I).contiguous(), beta, rho, q, 1.0, 0.0, u0, w0)
for C, K1, K2, BN, nw in ((128, 32, 32, 16, 1), (128, 32, 32, 8, 1), (128, 32, 32, 4, 1), (128, 32, 32, 16, 4), (128, 32, 64, 4, 1), (128, 32, 64, 8, 2)):
    Wpred = torch.cat([w0[None], Wp[C - 1:-1:C]], 0).contiguous()
    for rep in range(3):
        ph = {}
        A.alif_scan_forward(I, beta, rho, q, 1.0, 0.0, u0, w0, Wpred, 0.1 * q, C=C, K1=K1, K2=K2, BN=BN, nw=nw, BNc=1, D=4, phases=ph)
    print(f"C={C} K={K1}/{K2} BN={BN} nw={nw}: " + "  ".join(f"{k} {v:.2f}" for k, v in ph.items()) + f"  total {sum(ph.values()):.2f} ms", flush=True)
