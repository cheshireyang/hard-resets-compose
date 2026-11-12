"""Is the serial backward baseline tuned?  Old tiled kernel vs the prefetching one (LIF and ALIF), and the scan."""
import sys, os, math, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from lifscan import triton_lif as L, triton_alif as A
def tm(fn, reps=7):
    fn(); torch.cuda.synchronize(); ts = []
    for _ in range(reps):
        a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    return sorted(ts)[len(ts) // 2]
for N, T in ((16, 65536), (128, 65536), (128, 262144), (1024, 65536), (8192, 8192)):
    torch.manual_seed(0)
    I = (0.15 + 0.4 * torch.randn(T, N, device="cuda")).contiguous(); u0 = torch.zeros(N, device="cuda")
    U, S = L.serial_forward(I, 0.9, 1.0, 0.0, u0); GS = torch.randn(T, N, device="cuda"); GU = torch.zeros_like(GS)
    grid = [(BN, D, nw) for BN in (16, 32, 64, 128) for D in (4, 8) for nw in (1, 2) if BN <= max(16, N)]
    old = min(tm(lambda: L.serial_backward_tiled(I, u0, U, S, GS, GU, 0.9, 1.0, 0.0, BN=BN, D=D, num_warps=nw)) for BN, D, nw in grid)
    pf = min(tm(lambda: L.serial_backward_tiled_pf(I, u0, U, S, GS, GU, 0.9, 1.0, 0.0, BN=BN, D=D, num_warps=nw)) for BN, D, nw in grid)
    sc = min(tm(lambda: L.scan_backward_tiled(I, u0, U, S, GS, GU, 0.9, 1.0, 0.0, C=C, BNr=BNr, D=D)) for C in (32, 64, 128) for BNr in (16, 32) for D in (4, 8))
    ref = L.serial_backward_tiled(I, u0, U, S, GS, GU, 0.9, 1.0, 0.0); new = L.serial_backward_tiled_pf(I, u0, U, S, GS, GU, 0.9, 1.0, 0.0)
    print(f"LIF  N={N:5d} T={T:6d}: serial bwd old {old:7.2f}  prefetch {pf:7.2f} ({old / pf:4.2f}x faster, identical {torch.equal(ref, new)}) | scan {sc:7.2f} -> speedup old {old / sc:6.2f}x, vs prefetch {pf / sc:6.2f}x", flush=True)
    W = torch.zeros_like(I); Ua, Wa, Sa = A.alif_serial_forward(I * 0.3, 0.95, 0.995, 0.9, 1.0, 0.0, u0, u0, BN=32, D=8)
    Ia = (I * 0.3).contiguous()
    grid2 = [(BN, D, nw) for BN in (16, 32, 64) for D in (4, 8) for nw in (1, 2) if BN <= max(16, N)]
    old = min(tm(lambda: A.alif_backward(Ia, Ua, Wa, Sa, GS, 0.95, 0.995, 0.9, 1.0, u0, u0, mode="serial", BN=BN, D=D, nw=nw)) for BN, D, nw in grid2)
    pf = min(tm(lambda: A.alif_backward(Ia, Ua, Wa, Sa, GS, 0.95, 0.995, 0.9, 1.0, u0, u0, mode="serial_pf", BN=BN, D=D, nw=nw)) for BN, D, nw in grid2)
    sc = min(tm(lambda: A.alif_backward(Ia, Ua, Wa, Sa, GS, 0.95, 0.995, 0.9, 1.0, u0, u0, mode="scan", C=C, BN=BN, D=D)) for C in (32, 64, 128) for BN in (16, 32) for D in (4, 8) if BN <= max(16, N))
    r1 = A.alif_backward(Ia, Ua, Wa, Sa, GS, 0.95, 0.995, 0.9, 1.0, u0, u0, mode="serial"); r2 = A.alif_backward(Ia, Ua, Wa, Sa, GS, 0.95, 0.995, 0.9, 1.0, u0, u0, mode="serial_pf")
    print(f"ALIF N={N:5d} T={T:6d}: serial bwd old {old:7.2f}  prefetch {pf:7.2f} ({old / pf:4.2f}x faster, identical {torch.equal(r1[0], r2[0]) and torch.equal(r1[1], r2[1])}) | scan {sc:7.2f} -> speedup old {old / sc:6.2f}x, vs prefetch {pf / sc:6.2f}x", flush=True)
