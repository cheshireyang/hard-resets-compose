"""E14: fitting a hard-reset LIF neuron to a long spike recording by gradient descent.

Ground truth: hard-reset LIF with unknown leak beta, input gain g and bias b (theta = 1, u_r = 0),
driven by coloured-noise stimuli x_n(t); N trials of length T.  We fit three model classes
to the recorded spikes with a van Rossum loss (MSE between exponentially filtered spike trains):
  hard    : the exact hard-reset LIF (our layer)                       -- correctly specified
  soft    : the same neuron with subtractive reset (what soft-reset parallel methods train)
  noreset : the same neuron without reset (what PSN-style parallel methods train)
and report parameter recovery and spike-timing agreement on held-out stimuli.  On a GPU the
hard model can use the serial, scan or auto kernels; per-iteration time is recorded.

usage: python3 experiments/e14_fit.py DEVICE T N ITERS RESTARTS [modes...]
   e.g. python3 experiments/e14_fit.py cpu 8192 8 200 3 hard soft noreset
        python3 experiments/e14_fit.py cuda 262144 16 300 3 hard:serial hard:auto
"""
import sys, os, json, time, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from lifscan import fast
from lifscan.fit import lif_param_cpu
try:
    from lifscan.fit import lif_param
except Exception:
    lif_param = None

DEV, T, N, ITERS, RESTARTS = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
MODES = sys.argv[6:] or ["hard", "soft", "noreset"]
DT = torch.float32 if DEV == "cuda" else torch.float64
TRUE = dict(beta=0.95, gain=0.6, bias=0.04)
TAU_STIM, TAU_VR = 20.0, 10.0
torch.manual_seed(0)


def stimulus(T, N, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    w = torch.randn(T, N, generator=g, dtype=torch.float64)
    a = math.exp(-1.0 / TAU_STIM)
    # coloured noise via FFT filtering (exponential kernel), unit variance
    k = a ** torch.arange(T, dtype=torch.float64)
    x = torch.fft.irfft(torch.fft.rfft(w, n=2 * T, dim=0) * torch.fft.rfft(k, n=2 * T)[:, None], n=2 * T, dim=0)[:T]
    x = (x - x.mean(0)) / x.std(0)
    return x.to(DEV, DT)


def vr_filter(S):
    a = math.exp(-1.0 / TAU_VR)
    k = (a ** torch.arange(S.shape[0], device=S.device, dtype=S.dtype))
    return torch.fft.irfft(torch.fft.rfft(S, n=2 * S.shape[0], dim=0) * torch.fft.rfft(k, n=2 * S.shape[0])[:, None],
                           n=2 * S.shape[0], dim=0)[:S.shape[0]]


def hard_spikes(I, beta, mode):
    u0 = torch.zeros(I.shape[1], device=I.device, dtype=I.dtype)
    th = torch.tensor(1.0, device=I.device, dtype=I.dtype); ur = torch.tensor(0.0, device=I.device, dtype=I.dtype)
    if DEV == "cuda":
        return lif_param(I, u0, beta, th, ur, mode=mode)
    return lif_param_cpu(I, u0, beta, th, ur, C=64)


def soft_spikes(I, beta):
    # serial soft-reset LIF with surrogate gradient (autograd through the loop)
    u = torch.zeros(I.shape[1], device=I.device, dtype=I.dtype); out = []
    for t in range(I.shape[0]):
        v = beta * u + I[t]
        s = fast.SpikeFn.apply(v - 1.0, 4.0)
        u = v - s.detach()
        out.append(s)
    return torch.stack(out)


def noreset_spikes(I, beta):
    # leaky integration without reset: v = sum_j beta^(t-j) I_j  (FFT convolution), spikes = H(v - 1)
    k = beta ** torch.arange(I.shape[0], device=I.device, dtype=I.dtype)
    v = torch.fft.irfft(torch.fft.rfft(I, n=2 * I.shape[0], dim=0) * torch.fft.rfft(k, n=2 * I.shape[0])[:, None],
                        n=2 * I.shape[0], dim=0)[:I.shape[0]]
    return fast.SpikeFn.apply(v - 1.0, 4.0)


def model_spikes(kind, x, beta, gain, bias):
    I = gain * x + bias
    if kind.startswith("hard"):
        mode = kind.split(":")[1] if ":" in kind else "auto"
        return hard_spikes(I, beta, mode)
    if kind == "soft":
        return soft_spikes(I, beta)
    return noreset_spikes(I, beta)


def coincidence(Sm, St, win=2):
    """Fraction of target spikes matched by a model spike within +-win steps, and vice versa (F1)."""
    pad = torch.nn.functional.max_pool1d(Sm.T[:, None, :], 2 * win + 1, stride=1, padding=win)[:, 0, :].T
    padt = torch.nn.functional.max_pool1d(St.T[:, None, :], 2 * win + 1, stride=1, padding=win)[:, 0, :].T
    rec = (pad * St).sum() / St.sum().clamp(min=1); prec = (padt * Sm).sum() / Sm.sum().clamp(min=1)
    return float(2 * rec * prec / (rec + prec + 1e-12))


# data: training and held-out stimuli, ground-truth spikes
x_tr, x_te = stimulus(T, N, 1), stimulus(T, N, 2)
with torch.no_grad():
    b_true = torch.tensor(TRUE["beta"], device=DEV, dtype=DT)
    S_tr = model_spikes("hard:serial", x_tr, b_true, TRUE["gain"], TRUE["bias"]) if DEV == "cuda" else \
        model_spikes("hard", x_tr, b_true, TRUE["gain"], TRUE["bias"])
    S_te = model_spikes("hard:serial", x_te, b_true, TRUE["gain"], TRUE["bias"]) if DEV == "cuda" else \
        model_spikes("hard", x_te, b_true, TRUE["gain"], TRUE["bias"])
    Y_tr = vr_filter(S_tr)
print(f"data: T={T} N={N} rate={S_tr.mean().item():.4f}", flush=True)

results = dict(device=DEV, T=T, N=N, iters=ITERS, true=TRUE, rate=S_tr.mean().item(), runs=[])
for kind in MODES:
    for rs in range(RESTARTS):
        g = torch.Generator().manual_seed(100 + rs)
        init = dict(beta=0.80 + 0.18 * torch.rand(1, generator=g).item(), gain=0.3 + 0.7 * torch.rand(1, generator=g).item(),
                    bias=0.08 * torch.rand(1, generator=g).item())
        zb = torch.tensor(math.log(init["beta"] / (1 - init["beta"])), device=DEV, dtype=DT, requires_grad=True)
        gain = torch.tensor(init["gain"], device=DEV, dtype=DT, requires_grad=True)
        bias = torch.tensor(init["bias"], device=DEV, dtype=DT, requires_grad=True)
        opt = torch.optim.Adam([zb, gain, bias], lr=0.02)
        times = []; losses = []
        for it in range(ITERS):
            if DEV == "cuda": torch.cuda.synchronize()
            t0 = time.perf_counter()
            beta = torch.sigmoid(zb)
            S = model_spikes(kind, x_tr, beta, gain, bias)
            loss = ((vr_filter(S) - Y_tr) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            if DEV == "cuda": torch.cuda.synchronize()
            times.append(time.perf_counter() - t0); losses.append(loss.item())
        with torch.no_grad():
            beta = torch.sigmoid(zb)
            fit = dict(beta=beta.item(), gain=gain.item(), bias=bias.item())
            # evaluate every fitted model by simulating the *true* (hard-reset) dynamics with its parameters,
            # and also by its own dynamics
            S_own = model_spikes(kind, x_te, beta, gain, bias)
            S_hard = model_spikes("hard:serial" if DEV == "cuda" else "hard", x_te, beta, gain, bias)
        err = {k: abs(fit[k] - TRUE[k]) / abs(TRUE[k]) for k in TRUE}
        run = dict(kind=kind, restart=rs, init=init, fit=fit, rel_err=err, final_loss=losses[-1],
                   f1_own=coincidence(S_own, S_te), f1_hard=coincidence(S_hard, S_te),
                   ms_per_iter=1000 * sorted(times[3:])[len(times[3:]) // 2] if len(times) > 6 else 1000 * times[-1],
                   loss_curve=losses[::max(1, ITERS // 50)])
        results["runs"].append(run)
        print(f"{kind:12s} r{rs} fit beta={fit['beta']:.4f} gain={fit['gain']:.4f} bias={fit['bias']:.4f} | "
              f"rel err beta {err['beta']:.1e} gain {err['gain']:.1e} bias {err['bias']:.1e} | "
              f"F1 held-out own {run['f1_own']:.3f} | {run['ms_per_iter']:.1f} ms/iter", flush=True)
        os.makedirs("results", exist_ok=True)
        json.dump(results, open(f"results/e14_fit_{DEV}_T{T}_N{N}.json", "w"), indent=1)
