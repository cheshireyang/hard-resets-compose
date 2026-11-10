import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from lifscan import triton_lif as L
def timeit(fn, reps=25):
    fn(); torch.cuda.synchronize(); ts = []
    for _ in range(reps):
        a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    ts.sort(); return ts[len(ts) // 2]
for N, T in ((128, 1024), (1024, 1024), (8192, 1024), (1024, 2048), (8192, 2048)):
    I0 = (0.15 + 0.4 * torch.randn(T, N, device="cuda")).contiguous(); u0 = torch.zeros(N, device="cuda"); G = torch.randn(T, N, device="cuda")
    res = []
    for rep in range(3):
        for bwd in ("serial", "scan"):
            I = I0.clone().requires_grad_(True)
            def step():
                I.grad = None; s = L.LIFAuto.apply(I, u0, 0.9, 1.0, 0.0, 4.0, 16, 0, "serial", bwd); (s * G).sum().backward()
            res.append((bwd, round(timeit(step), 3)))
    print(N, T, res, flush=True)
