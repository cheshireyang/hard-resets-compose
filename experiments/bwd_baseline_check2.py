import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from lifscan import triton_lif as L, triton_alif as A
def tm(fn, reps=5):
    fn(); torch.cuda.synchronize(); ts = []
    for _ in range(reps):
        a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    return sorted(ts)[len(ts) // 2]
for N, T in ((16, 65536), (128, 65536), (8192, 8192)):
    torch.manual_seed(0)
    I = (0.15 + 0.4 * torch.randn(T, N, device="cuda")).contiguous(); u0 = torch.zeros(N, device="cuda")
    U, S = L.serial_forward(I, 0.9, 1.0, 0.0, u0); GS = torch.randn(T, N, device="cuda"); GU = torch.zeros_like(GS)
    fw = {}
    for BN in (16, 32, 64, 128):
        if BN > max(16, N): continue
        for D in (4, 8, 16, 32):
            for nw in (1, 2, 4):
                fw[(BN, D, nw)] = tm(lambda: L.serial_forward_tiled(I, 0.9, 1.0, 0.0, u0, BN=BN, D=D, num_warps=nw))
    res = {}
    for BN in (16, 32, 64, 128):
        if BN > max(16, N): continue
        for D in (4, 8, 16, 32):
            for nw in (1, 2, 4):
                res[(BN, D, nw)] = tm(lambda: L.serial_backward_tiled_pf(I, u0, U, S, GS, GU, 0.9, 1.0, 0.0, BN=BN, D=D, num_warps=nw))
    kb = min(res, key=res.get); kf = min(fw, key=fw.get)
    print(f"LIF N={N} T={T}: serial fwd best {fw[kf]:.2f} at {kf} | serial bwd(pf) best {res[kb]:.2f} at {kb}; by D: " +
          ", ".join(f"D={D}:{min(v for k, v in res.items() if k[1] == D):.2f}" for D in (4, 8, 16, 32)), flush=True)
