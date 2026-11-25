"""Quick phase profile of the ALIF scan on one cell (development aid)."""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, triton
from lifscan import triton_alif as A, triton_lif as L

def tm(fn, reps=7):
    fn(); torch.cuda.synchronize(); ts = []
    for _ in range(reps):
        a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    return sorted(ts)[len(ts) // 2]

name, beta, rho, q, mfree = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
N, T = int(sys.argv[6]), int(sys.argv[7])
torch.manual_seed(0)
I = ((1 - beta) * mfree + 0.5 * math.sqrt(1 - beta * beta) * torch.randn(T, N, device="cuda")).contiguous()
u0 = torch.zeros(N, device="cuda"); w0 = torch.zeros(N, device="cuda")
ser = {}
for BN in (16, 32, 64):
    for D in (0, 4, 8):
        if BN > N and BN != 16: continue
        ser[(BN, D)] = tm(lambda: A.alif_serial_forward(I, beta, rho, q, 1.0, 0.0, u0, w0, BN=BN, D=D))
bs = min(ser, key=ser.get); print(f"serial best {ser[bs]:.2f} ms at BN,D={bs}  (all: " + ", ".join(f"{k}:{v:.2f}" for k, v in ser.items()) + ")")
_, Wr, Sr = A.alif_serial_forward(I, beta, rho, q, 1.0, 0.0, u0, w0)
_, Wp, _ = A.alif_serial_forward((1.005 * I).contiguous(), beta, rho, q, 1.0, 0.0, u0, w0)
for C in (64, 128, 256):
    M = T // C
    Wpred = torch.cat([w0[None], Wp[C - 1:-1:C]], 0).contiguous()
    for dwq in (0.03, 0.1, 0.3):
        for K1, K2 in ((16, 32), (32, 32), (32, 64)):
            for BNc in (1, 4):
                st = {}
                f = lambda: A.alif_scan_forward(I, beta, rho, q, 1.0, 0.0, u0, w0, Wpred, dwq * q, C=C, K1=K1, K2=K2, BN=16, BNc=BNc, D=4)
                t = tm(f); A.alif_scan_forward(I, beta, rho, q, 1.0, 0.0, u0, w0, Wpred, dwq * q, C=C, K1=K1, K2=K2, BN=16, BNc=BNc, D=4, stats=st)
                print(f"C={C:3d} dw={dwq:4.2f}q K={K1:2d}/{K2:2d} BNc={BNc}: {t:7.2f} ms  speedup {ser[bs] / t:5.2f}x  miss {100 * st['miss_frac']:5.1f}% ovf {100 * st['overflow_frac']:4.1f}% pieces {st['mean_pieces']:5.1f} lines {st['mean_lines']:5.1f}", flush=True)
