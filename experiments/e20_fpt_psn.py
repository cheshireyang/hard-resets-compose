"""E20: head-to-head with FPT (ICML 2025) and PSN.

FPT neuron re-implemented following the official code (github.com/WanjinVon/FPT, FPT_adaptive_alpha):
dense lower-triangular T x T matrix Lambda[i, j] = lam^(i-j), threshold bias -1, three sigmoid-relaxed
fixed-point iterations with sharpening alpha, 3*alpha, 12*alpha (alpha = 1), subtractive reset by the
relaxed spike, and a final Bernoulli sample.  Its target dynamics are soft-reset LIF with leak lam and
threshold 1, so we compare it with our exact soft-reset scan (and the serial soft-reset kernel).
PSN (no reset, dense T x T) is timed as a reference.
Reports: forward+backward time, and the fraction of spikes that differ from exact soft-reset dynamics
(Bernoulli output, and the deterministic variant that thresholds the last relaxed iterate at 0.5)."""
import sys, os, json, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from lifscan import triton_soft as TS
from lifscan.fit import lif_soft_scan


class SurrogateSigmoid(torch.autograd.Function):          # as in the official FPT code
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x); ctx.alpha = alpha
        return torch.sigmoid(x * alpha)

    @staticmethod
    def backward(ctx, g):
        ab = max(1.0, ctx.alpha / 3)
        s = torch.sigmoid(ctx.saved_tensors[0] * ab)
        return g * (1 - s) * s * ab, None


class Bernoulli(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p):
        return torch.bernoulli(p)

    @staticmethod
    def backward(ctx, g):
        return g


def lam_matrix(T, lam, dev):
    i = torch.arange(T, device=dev)
    d = (i[:, None] - i[None, :]).float()
    return torch.where(d >= 0, lam ** d.clamp(min=0), torch.zeros((), device=dev))


def fpt_forward(x, Lam, alpha=1.0, deterministic=False):
    # x: [T, N]
    h = Lam @ x - 1.0
    s = SurrogateSigmoid.apply(h, alpha)
    h = Lam @ (x - s) + s - 1.0
    s = SurrogateSigmoid.apply(h, alpha * 3)
    h = Lam @ (x - s) + s - 1.0
    p = SurrogateSigmoid.apply(h, alpha * 12)
    return (p >= 0.5).float() if deterministic else Bernoulli.apply(p)


def psn_forward(x, Lam):
    h = Lam @ x - 1.0
    return SurrogateSigmoid.apply(h, 4.0)


def timeit(fn, reps=5):
    fn(); torch.cuda.synchronize(); ts = []
    for _ in range(reps):
        a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    ts.sort(); return ts[len(ts) // 2]


rows = []
for lam in (0.5, 0.9):
    for N, T in itertools.product((128, 1024), (1024, 4096, 16384, 32768)):
        torch.manual_seed(0)
        x = (0.15 + 0.4 * torch.randn(T, N, device="cuda")).contiguous()
        u0 = torch.zeros(N, device="cuda")
        G = torch.randn(T, N, device="cuda")
        r = dict(lam=lam, N=N, T=T)
        _, S_exact = TS.soft_serial_forward(x, lam, 1.0, u0)
        r["rate_exact"] = S_exact.mean().item()
        # ours: exact soft scan forward + FFT adjoint, and the serial soft kernels
        C = 256 if T >= 16384 else (128 if T >= 4096 else 64)
        for mode in ("scan", "serial"):
            xi = x.clone().requires_grad_(True)
            def step():
                xi.grad = None
                S = lif_soft_scan(xi, u0, torch.tensor(lam, device="cuda"), mode=mode, C=C, K=32)
                (S * G).sum().backward()
            r[f"ms_soft_{mode}"] = timeit(step)
        S_scan = lif_soft_scan(x, u0, torch.tensor(lam, device="cuda"), mode="scan", C=C, K=32)
        r["mismatch_scan"] = (S_scan != S_exact).float().mean().item()
        # FPT and PSN (dense T x T; skip when the matrix does not fit)
        if T * T * 4 * 3 < 20e9:
            Lam = lam_matrix(T, lam, "cuda")
            xi = x.clone().requires_grad_(True)
            def fstep():
                xi.grad = None; S = fpt_forward(xi, Lam); (S * G).sum().backward()
            r["ms_fpt"] = timeit(fstep)
            def pstep():
                xi.grad = None; S = psn_forward(xi, Lam); (S * G).sum().backward()
            r["ms_psn"] = timeit(pstep)
            with torch.no_grad():
                torch.manual_seed(1)
                Sb = fpt_forward(x, Lam); Sd = fpt_forward(x, Lam, deterministic=True)
            r["mismatch_fpt_bernoulli"] = (Sb != S_exact).float().mean().item()
            r["mismatch_fpt_det"] = (Sd != S_exact).float().mean().item()
            r["rate_fpt_det"] = Sd.mean().item()
            del Lam
        else:
            r["ms_fpt"] = r["ms_psn"] = None
        torch.cuda.empty_cache()
        rows.append(r)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
        json.dump(rows, open("results/e20_fpt_psn.json", "w"), indent=1)
