"""E20b: sanity check of the FPT re-implementation of E20 (CPU, float64).

FPT iterates s <- H_alpha(Lambda (x - s) + s - 1), whose fixed points with a hard Heaviside are exactly the
soft-reset LIF spike trains.  If the re-implementation is right, then
  (1) with a hard threshold the iteration reaches the exact spikes after finitely many rounds (at most T), and
  (2) the official three-round sigmoid schedule (alpha = 1, 3, 12) is accurate where membrane memory is short
      (lam = 0.5) and inaccurate at lam = 0.9, where the first rounds overshoot.
We also record the firing rate after every round, which shows where FPT's high rate at lam = 0.9 comes from."""
import json, math
import torch

torch.set_default_dtype(torch.float64)


def lam_matrix(T, lam):
    i = torch.arange(T)
    d = (i[:, None] - i[None, :]).double()
    return torch.where(d >= 0, lam ** d.clamp(min=0), torch.zeros(()))


def exact_soft(x, lam):
    u = torch.zeros(x.shape[1]); S = torch.zeros_like(x)
    for t in range(x.shape[0]):
        v = lam * u + x[t]; s = (v >= 1.0).double(); u = v - s; S[t] = s
    return S


def fpt_rounds(x, Lam, alphas, hard_last=False):
    """Returns the spike estimate after each round (official code: s0 = sig(alpha0 h(0)), then refine)."""
    out = []
    h = Lam @ x - 1.0
    s = torch.sigmoid(alphas[0] * h) if math.isfinite(alphas[0]) else (h >= 0).double()
    out.append(s)
    for a in alphas[1:]:
        h = Lam @ (x - s) + s - 1.0
        s = torch.sigmoid(a * h) if math.isfinite(a) else (h >= 0).double()
        out.append(s)
    return out


rows = []
torch.manual_seed(0)
T, N = 1024, 128
for lam in (0.5, 0.9):
    x = 0.15 + 0.4 * torch.randn(T, N)
    S = exact_soft(x, lam)
    Lam = lam_matrix(T, lam)
    r = dict(lam=lam, T=T, N=N, rate_exact=S.mean().item())
    # (a) official schedule, three rounds; Bernoulli and thresholded output
    p = fpt_rounds(x, Lam, [1.0, 3.0, 12.0])
    torch.manual_seed(1)
    r["official_bernoulli_err"] = (torch.bernoulli(p[-1]) != S).double().mean().item()
    r["official_det_err"] = ((p[-1] >= 0.5).double() != S).double().mean().item()
    r["official_rate_per_round"] = [q.mean().item() for q in p]
    # (b) hard-threshold fixed-point iteration: rounds until exact
    s = (Lam @ x - 1.0 >= 0).double(); errs = [(s != S).double().mean().item()]
    for k in range(1, T + 1):
        s_new = (Lam @ (x - s) + s - 1.0 >= 0).double()
        errs.append((s_new != S).double().mean().item())
        if torch.equal(s_new, s): break
        s = s_new
    r["hard_rounds_to_fixed_point"] = len(errs) - 1
    r["hard_final_err"] = errs[-1]
    r["hard_err_first10"] = errs[:10]
    # (c) official schedule extended with alpha = 12 for more rounds (thresholded)
    ext = {}
    for K in (3, 5, 10, 20, 50):
        q = fpt_rounds(x, Lam, [1.0, 3.0] + [12.0] * (K - 2))
        ext[K] = ((q[-1] >= 0.5).double() != S).double().mean().item()
    r["extended_det_err"] = ext
    rows.append(r)
    print(json.dumps(r), flush=True)
json.dump(rows, open("results/e20b_fpt_sanity.json", "w"), indent=1)
