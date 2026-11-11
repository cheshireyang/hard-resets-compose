"""E19: exact fixed-point (Picard) iteration on the spike pattern -- the natural *iterative* exact
baseline (FPT/DEER-style, but without relaxation).  Given spikes s, the hard-reset recurrence is
affine (u_t = beta*(1-s_t)*u_{t-1} + (1-s_t)*I_t + s_t*u_r) and is solved by an affine scan; spikes are
then recomputed from v_t = beta*u_{t-1} + I_t.  We count iterations to the exact fixed point and time
it against the non-iterative map scan (v3) and the tuned serial kernel.  Also: spike error after
K = 3 iterations (the budget used by FPT)."""
import sys, os, json, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from lifscan import triton_lif as L

def affine_scan_fwd(a, b, u0):
    """u_t = a_t u_{t-1} + b_t (Hillis-Steele, log T passes)."""
    a = a.clone(); b = b.clone()
    b[0] = a[0] * u0 + b[0]; a[0] = 0
    T = a.shape[0]; d = 1
    while d < T:
        ap = torch.ones_like(a); bp = torch.zeros_like(b)
        ap[d:] = a[:-d]; bp[d:] = b[:-d]
        b = a * bp + b; a = a * ap
        d *= 2
    return b

def picard(I, beta, theta, u_r, u0, max_iter=100000, record_k=3):
    s = torch.zeros_like(I)
    it = 0; s_k = None
    while it < max_iter:
        it += 1
        a = beta * (1 - s); b = (1 - s) * I + s * u_r
        u = affine_scan_fwd(a, b, u0)
        uprev = torch.cat([u0[None], u[:-1]], 0)
        s_new = ((beta * uprev + I) >= theta).to(I.dtype)
        if it == record_k: s_k = s_new.clone()
        if torch.equal(s_new, s): break
        s = s_new
    return u, s, it, s_k

def timeit(fn, reps=3):
    fn(); torch.cuda.synchronize(); ts = []
    for _ in range(reps):
        a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    ts.sort(); return ts[len(ts) // 2]

rows = []
beta = 0.9
for N, T in itertools.product((128, 1024), (1024, 4096, 16384, 65536)):
    for mu in (0.05, 0.15):
        torch.manual_seed(0)
        I = (mu + 0.4 * torch.randn(T, N, device="cuda")).contiguous(); u0 = torch.zeros(N, device="cuda")
        Us, Ss = L.serial_forward_tiled(I, beta, 1.0, 0.0, u0, BN=32, D=8, num_warps=1)
        u, s, iters, s3 = picard(I, beta, 1.0, 0.0, u0)
        exact = torch.equal(s, Ss)
        err3 = (s3 != Ss).float().mean().item() if s3 is not None else 0.0
        t_pic = timeit(lambda: picard(I, beta, 1.0, 0.0, u0), reps=1)
        t_ser = timeit(lambda: L.serial_forward_tiled(I, beta, 1.0, 0.0, u0, BN=32, D=8, num_warps=1))
        t_scan = timeit(lambda: L.scan_forward_v3(I, beta, 1.0, 0.0, u0, C=L._pick_C(T), K=16))
        r = dict(N=N, T=T, mu=mu, rate=Ss.mean().item(), iters=iters, exact=exact, err_after3=err3,
                 ms_picard=t_pic, ms_serial=t_ser, ms_scan=t_scan, spikes_per_neuron=Ss.sum(0).mean().item())
        rows.append(r)
        print(f"N={N:<5} T={T:<6} rate={r['rate']:.3f} | Picard iters {iters:6d} (exact={exact}) err@3 {err3:.3f} | "
              f"ms: picard {t_pic:9.2f} serial {t_ser:7.3f} scan {t_scan:7.3f}", flush=True)
        json.dump(rows, open("results/e19_picard.json", "w"), indent=1)
