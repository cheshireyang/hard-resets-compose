import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from lifscan import triton_soft as S
def timeit(fn, reps=7):
    fn(); torch.cuda.synchronize(); ts = []
    for _ in range(reps):
        a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    ts.sort(); return ts[len(ts)//2]
for N, T in ((4, 65536), (128, 65536), (1024, 16384)):
    I = (0.15 + 0.4 * torch.randn(T, N, device="cuda")).contiguous(); u0 = torch.zeros(N, device="cuda")
    Ur, Sr = S.soft_serial_forward(I, 0.95, 1.0, u0)
    ts = min(timeit(lambda: S.soft_serial_forward(I, 0.95, 1.0, u0)), timeit(lambda: S.soft_serial_forward_tiled(I, 0.95, 1.0, u0)))
    for C, K, BN in ((32, 32, 16), (64, 32, 16), (32, 16, 16), (16, 16, 16)):
        st = {}; _, Sp = S.soft_scan_forward(I, 0.95, 1.0, u0, C=C, K=K, BN=BN, stats=st)
        U2, S2 = S.soft_scan_forward(I, 0.95, 1.0, u0, C=C, K=K, BN=BN, repair=10**6)
        t = timeit(lambda: S.soft_scan_forward(I, 0.95, 1.0, u0, C=C, K=K, BN=BN))
        print(f"N={N} T={T} C={C} K={K}: serial {ts:.3f} scan {t:.3f} x{ts/t:.2f} ovf {st['overflow_frac']:.3f} bit {torch.equal(S2, Sr) and torch.equal(U2, Ur)}", flush=True)
