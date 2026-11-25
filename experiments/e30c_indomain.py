"""E30c: in-domain (pruned) size of hard-reset segment maps vs. theory (CPU, float64).

Theorem (in-domain size).  For a window [l, theta] with l independent of the segment's inputs,
  #breakpoints in [l, theta] = #{k <= L : k is a record and a_k >= l}
and, by time reversal,
  E[# | l] = sum_{k=1}^{L} P(W_m > theta(1-beta^m) for m = 1..k,  W_k < theta - l beta^k),   W_m = sum_{i<=m} beta^{i-1} I_i.
Records below theta are the strict descending ladder epochs of the walk a_k (a_0 = theta), whose increments after a
ladder epoch k_j are beta^{-k_j} times a fresh copy; hence the ladder heights dominate beta^{-(j-1)} H_j with H_j i.i.d.
copies of the first ladder height H of the walk with increments ((1-beta)theta - I_i)/beta^i, and
  E[# | l] <= U_beta(w) := sum_{n>=1} P(sum_{j=1}^n beta^{1-j} H_j <= w),   w = theta - l,
with equality as L -> inf for beta = 1, where U_1 is the renewal function of the strict ascending ladder heights of S.
For beta = 1: U_1(w) ~ w / E[H];  E[H] = sigma/sqrt(2) for symmetric inputs (Gaussian: U_1(w) = sqrt(2) w/sigma - 0.176
+ o(1)), and E[H] = mu / pi_inf for drift mu > 0.

This script (1) regenerates the inputs of E1c exactly (same seed and order), counts records in the envelope window
directly and compares with E1c's map-builder counts, (2) checks the time-reversal identity with independent inputs,
(3) evaluates the bound U_beta(w) and the density bound of Prop. 'domain' by Monte Carlo."""
import sys, os, json, math
import numpy as np
import torch
from scipy.stats import norm

TH = 1.0
T, N = 16384, 64
LS = (16, 64, 256, 1024, 4096)
rng = np.random.default_rng(12345)


def envelope(I, beta):
    l = np.empty((I.shape[0] + 1, I.shape[1])); l[0] = 0.0
    for t in range(I.shape[0]):
        l[t + 1] = np.minimum(0.0, beta * l[t] + I[t])
    return l


def records_in_window(Ic, beta, lo):
    """Ic: [L, n] chunk inputs, lo: [n]; returns #records k with lo <= a_k <= theta (Theorem 1), per column."""
    L, n = Ic.shape
    b = np.zeros(n); m = np.full(n, np.inf); cnt = np.zeros(n, dtype=np.int64)
    for k in range(1, L + 1):
        b = beta * b + Ic[k - 1]
        a = (TH - b) / beta ** k
        rec = a < m
        cnt += rec & (a >= lo) & (a <= TH)
        m = np.minimum(m, a)
    return cnt


def reversal_mc(beta, mu, sd, L, lo_samples, reps):
    """E[#] by the time-reversal formula, with fresh inputs, for l drawn from lo_samples."""
    lo = rng.choice(lo_samples, size=reps)
    W = np.zeros(reps); alive = np.ones(reps, dtype=bool); tot = np.zeros(reps)
    for k in range(1, L + 1):
        W = W + beta ** (k - 1) * (mu + sd * rng.standard_normal(reps))
        alive &= W > TH * (1 - beta ** k)
        tot += alive & (W < TH - lo * beta ** k)
        if not alive.any(): break
    return tot.mean(), tot.std() / math.sqrt(reps)


def ladder_heights(beta, mu, sd, n, kmax):
    """First strict descending ladder height of Y_i = sum_{r<=i} ((1-beta)theta - I_r)/beta^r (beta = 1: ascending of S)."""
    H = np.full(n, np.inf); Y = np.zeros(n); open_ = np.ones(n, dtype=bool)
    for i in range(1, kmax + 1):
        idx = np.nonzero(open_)[0]
        if len(idx) == 0: break
        Y[idx] += ((1 - beta) * TH - (mu + sd * rng.standard_normal(len(idx)))) / beta ** i
        done = idx[Y[idx] < 0]
        H[done] = -Y[done]; open_[done] = False
        if beta == 1.0 and mu < 0:                    # far below 0 with negative drift: returns with P < e^-40
            open_[idx[Y[idx] > 20 * sd * sd / -mu]] = False
        if beta < 1 and beta ** -i > 1e250: break
    return H, open_.mean()


def U_beta(w_samples, H, beta, reps=20000, nmax=400, q_def=None):
    """E over w of sum_{n>=1} P(sum_{j<=n} beta^{1-j} H_j <= w), H_j drawn from the simulated ladder heights.
    beta < 1: unresolved walks count as H = inf (their eventual heights exceed 1e8).  beta = 1: H = inf with the exact
    defect probability q_def (Sparre Andersen), otherwise drawn from the resolved (finite) heights."""
    w = rng.choice(w_samples, size=reps)
    Hf = H[np.isfinite(H)]
    s = np.zeros(reps); tot = np.zeros(reps); alive = np.ones(reps, dtype=bool)
    for j in range(1, nmax + 1):
        if q_def is None:
            h = rng.choice(H, size=reps)
        else:
            h = np.where(rng.random(reps) < q_def, np.inf, rng.choice(Hf, size=reps))
        s = s + beta ** (1 - j) * h
        alive &= s <= w
        tot += alive
        if not alive.any(): break
    return tot.mean(), tot.std() / math.sqrt(reps)


def defect(mu, sd, kmax=2000000):
    """P(no strict ascending ladder epoch) = exp(-sum_k P(S_k > 0)/k) for Gaussian steps; 0 if mu >= 0."""
    if mu >= 0: return 0.0
    k = np.arange(1, kmax + 1)
    return math.exp(-np.sum(norm.cdf(mu * np.sqrt(k) / sd) / k))


def pi_inf(mu, sd, kmax=100000):
    k = np.arange(1, kmax + 1)
    return math.exp(-np.sum(norm.cdf(-mu * np.sqrt(k) / sd) / k))


e1c = {(r["beta"], r["mu"], r["sd"], r["L"]): r for r in json.load(open("results/e1c_tight.json"))}
NBIG = int(os.environ.get("E30C_NBIG", "512"))
torch.manual_seed(0)                      # identical input stream to E1c
rows = []
for beta in (0.9, 0.99, 0.999, 1.0):
    for mu, sd in ((0.15, 0.3), (0.0, 0.3), (-0.1, 0.5)):
        I = (mu + sd * torch.randn(T, N, dtype=torch.float64)).numpy()
        l = envelope(I, beta)
        Ib = mu + sd * rng.standard_normal((T, NBIG)); lb = envelope(Ib, beta)       # independent large sample
        H, unres = ladder_heights(beta, mu, sd, 400000, 200000 if beta == 1.0 else 20000)
        q_def = defect(mu, sd) if beta == 1.0 else None
        rho = 1.0 / (sd * math.sqrt(2 * math.pi))
        for L in LS:
            M = T // L
            cnt = np.concatenate([records_in_window(I[m * L:(m + 1) * L], beta, l[m * L]) for m in range(M)])
            cb = np.concatenate([records_in_window(Ib[m * L:(m + 1) * L], beta, lb[m * L]) for m in range(M)])
            w = TH - np.concatenate([lb[m * L] for m in range(M)])
            r = dict(beta=beta, mu=mu, sd=sd, L=L, e1c_repro=float(1 + cnt.mean()), e1c=float(e1c[(beta, mu, sd, L)]["tight_mean"]),
                     measured=float(1 + cb.mean()), measured_se=float(cb.std() / math.sqrt(len(cb))), n_chunks=int(len(cb)),
                     w_mean=float(w.mean()), w_p90=float(np.quantile(w, 0.9)), H_unresolved=float(unres), q_def=q_def)
            mc, se = reversal_mc(beta, mu, sd, L, TH - w, 40000 if L <= 1024 else 10000)
            r["reversal_mc"], r["reversal_se"] = 1 + mc, se
            u, use = U_beta(w, H, beta, reps=200000, nmax=2000, q_def=q_def)
            r["U_bound"], r["U_se"] = 1 + u, use
            r["z_excess"] = (r["measured"] - r["U_bound"]) / math.sqrt(r["measured_se"] ** 2 + use ** 2)
            r["density_bound"] = 1 + rho * w.mean() * beta / (1 - beta) if beta < 1 else math.inf
            if beta == 1.0 and mu == 0:
                r["asym"] = 1 + float(np.mean(np.sqrt(2) * w / sd - 0.176))
            elif beta == 1.0 and mu > 0:
                r["asym"] = 1 + float(np.mean(pi_inf(mu, sd) * w / mu))
            rows.append(r)
            print(f"beta={beta:<6}mu={mu:<5}L={L:<5} E1c {r['e1c']:6.2f} repro {r['e1c_repro']:6.2f} | measured {r['measured']:6.2f}+-{r['measured_se']:.2f} "
                  f"| reversal {1 + mc:6.2f}+-{se:.2f} | U-bound {r['U_bound']:7.2f}+-{r['U_se']:.2f} z {r['z_excess']:5.2f} density {r['density_bound']:8.1f} asym {r.get('asym', float('nan')):6.2f} | "
                  f"w mean {w.mean():6.2f}", flush=True)
            json.dump(rows, open("results/e30c_indomain.json", "w"), indent=1)

# U_1 for Gaussian zero drift vs the asymptotic sqrt(2) w / sigma - 0.176 (sigma = 1)
H, unres = ladder_heights(1.0, 0.0, 1.0, 200000, 200000); Hf = H[np.isfinite(H)]
chk = []
for wv in (0.5, 1, 2, 5, 10, 20):
    u, _ = U_beta(np.array([wv]), H, 1.0, reps=40000, nmax=2000, q_def=0.0)
    chk.append(dict(w=wv, U=u, asym=math.sqrt(2) * wv - 0.176))
    print(f"zero drift, sigma=1: w={wv:<5} U={u:7.3f}  sqrt(2)w-0.176={math.sqrt(2) * wv - 0.176:7.3f}  (E[H]={Hf.mean():.4f}, 1/sqrt2={1 / math.sqrt(2):.4f})")
json.dump(dict(rows=rows, U_check=chk, EH=float(Hf.mean()), unresolved=float(unres)), open("results/e30c_indomain.json", "w"), indent=1)
