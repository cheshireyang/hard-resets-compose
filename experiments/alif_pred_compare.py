"""Miss rates of the ALIF box scan for different predictors of the chunk-entry adaptation (development aid)."""
import sys, os, math, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from lifscan import triton_alif as A
for (name, beta, rho, q), mfree in itertools.product((("lsnn", 0.95, 0.995, 0.9), ("appD-b", 0.99, 0.99, 0.25), ("glif2", 0.99, 0.96, 0.2)), (1.0, 2.0)):
    torch.manual_seed(0); T, N, C = 65536, 128, 128
    I = ((1 - beta) * mfree + 0.5 * math.sqrt(1 - beta * beta) * torch.randn(T, N, device="cuda")).contiguous()
    u0 = torch.zeros(N, device="cuda"); w0 = torch.zeros(N, device="cuda")
    for eps in (0.005, 0.02):
        Up, Wp, _ = A.alif_serial_forward(((1 + eps) * I).contiguous(), beta, rho, q, 1.0, 0.0, u0, w0)
        UI = torch.cat([u0[None], Up[C - 1:-1:C]], 0).contiguous(); WI = torch.cat([w0[None], Wp[C - 1:-1:C]], 0).contiguous()
        preds = {"prev": WI, "warm4": A.alif_warm_predict(I, beta, rho, q, 1.0, 0.0, u0, w0, C, Wn=4)}
        for Wn in (1, 2):
            preds[f"prev+warm{Wn}"] = A.alif_warm_predict(I, beta, rho, q, 1.0, 0.0, u0, w0, C, Wn=Wn, UI=UI, WI=WI)
        line = f"{name:6s} m={mfree} eps={eps}:"
        for k, Wpr in preds.items():
            for dwq in (0.1, 0.3):
                st = {}
                A.alif_scan_forward(I, beta, rho, q, 1.0, 0.0, u0, w0, Wpr.contiguous(), dwq * q, C=C, K1=32, K2=32, BN=4, stats=st)
                line += f"  {k}@{dwq}: {100 * st['miss_frac']:4.1f}%"
        print(line, flush=True)
