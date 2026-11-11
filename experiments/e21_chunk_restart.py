"""E21: a discrete-time version of the Bullet-Trains idea (scan the free trajectory over a chunk,
commit the first output spike, restart right after it) vs our map scan and the tuned serial kernel,
as a function of the firing rate.  The chunk-restart method is exact but its number of sequential
rounds grows with the number of spikes; the map scan's depth does not."""
import sys, os, json, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, triton, triton.language as tl
from lifscan import triton_lif as L


@triton.jit
def chunk_restart_kernel(I, U0, Uout, Sout, T, N, beta, theta, u_r, C: tl.constexpr):
    n = tl.program_id(0)
    j = tl.arange(0, C)
    bpow = tl.exp(j.to(tl.float32) * tl.log(beta))          # beta^j
    binv = 1.0 / bpow                                        # beta^-j
    u = tl.load(U0 + n)
    t = 0
    rounds = 0
    while t < T:
        rounds += 1
        idx = t + j
        valid = idx < T
        x = tl.load(I + idx * N + n, mask=valid, other=0.0)
        # free trajectory v_{t+j} = beta^{j+1} u + sum_{i<=j} beta^{j-i} x_i
        cs = tl.cumsum(x * binv, axis=0)
        v = bpow * (beta * u + cs)
        hit = (v >= theta) & valid
        first = tl.min(tl.where(hit, j, C), axis=0)          # first spike in the chunk (C if none)
        nlast = tl.minimum(first, tl.minimum(C - 1, T - 1 - t))
        upto = j <= nlast
        uo = tl.where(j == first, u_r, v)
        tl.store(Uout + idx * N + n, uo, mask=upto & valid)
        tl.store(Sout + idx * N + n, (j == first).to(tl.float32), mask=upto & valid)
        u = tl.sum(tl.where(j == nlast, uo, 0.0), axis=0)
        t = t + nlast + 1


def chunk_restart(I, beta, theta, u_r, u0, C=64):
    T, N = I.shape
    U = torch.empty_like(I); S = torch.empty_like(I)
    chunk_restart_kernel[(N,)](I, u0, U, S, T, N, beta, theta, u_r, C=C, num_warps=1)
    return U, S


def timeit(fn, reps=5):
    fn(); torch.cuda.synchronize(); ts = []
    for _ in range(reps):
        a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    ts.sort(); return ts[len(ts) // 2]


rows = []
beta, T = 0.9, 65536
for N in (128, 1024):
    for mu in (-0.05, 0.05, 0.15, 0.3, 0.5):
        torch.manual_seed(0)
        I = (mu + 0.4 * torch.randn(T, N, device="cuda")).contiguous(); u0 = torch.zeros(N, device="cuda")
        Us, Ss = L.serial_forward_tiled(I, beta, 1.0, 0.0, u0, BN=32, D=8, num_warps=1)
        r = dict(N=N, T=T, mu=mu, rate=Ss.mean().item())
        best = None
        for C in (32, 64, 128):
            Uc, Sc = chunk_restart(I, beta, 1.0, 0.0, u0, C=C)
            t = timeit(lambda: chunk_restart(I, beta, 1.0, 0.0, u0, C=C))
            if best is None or t < best[0]:
                best = (t, C, (Sc != Ss).float().mean().item())
        r["ms_chunk_restart"], r["chunk_C"], r["mismatch_chunk_restart"] = best
        r["ms_serial"] = min(timeit(lambda: L.serial_forward_tiled(I, beta, 1.0, 0.0, u0, BN=32, D=8, num_warps=1)),
                             timeit(lambda: L.serial_forward(I, beta, 1.0, 0.0, u0)))
        r["ms_scan"] = min(timeit(lambda: L.scan_forward_v3(I, beta, 1.0, 0.0, u0, C=C2, K=K))
                           for C2 in (128, 256) for K in (8, 16))
        rows.append(r)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
        json.dump(rows, open("results/e21_chunk_restart.json", "w"), indent=1)
