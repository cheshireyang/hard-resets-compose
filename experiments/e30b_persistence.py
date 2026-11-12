"""E30b: checks of the general characterisation of record counts.
(b) beta = 1, non-symmetric inputs: E[1+R_L] = 1 + sum_{n<L} pi_n with sum pi_n s^n = exp(sum_k s^k P(S_k > 0) / k)
    (Sparre Andersen).  For I = Exp(1) - 1, S_k = Gamma(k,1) - k, so P(S_k > 0) = Q(k, k) exactly.
(c) beta < 1: E[R_L] = sum_{k<=L} pi_{k-1}, pi_n = P(W_m > theta (1 - beta^m), m = 1..n), W_m = sum_{i<=m} beta^(i-1) I_i;
    records counted directly vs. persistence probabilities estimated independently; linear growth rate pi_inf."""
import sys, os, json, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.special import gammaincc

out = {}
# (b) exact generating function for Exp(1) - 1
K = 4096
pk = np.array([gammaincc(k, k) for k in range(1, K + 1)])          # P(S_k > 0) = P(Gamma(k) > k)
# coefficients of exp(sum_k pk s^k / k): c_0 = 1, n c_n = sum_{k=1}^n k a_k c_{n-k} with a_k = pk/k
a = pk / np.arange(1, K + 1)
c = np.zeros(K + 1); c[0] = 1.0
for n in range(1, K + 1):
    c[n] = np.dot(np.arange(1, n + 1) * a[:n], c[n - 1::-1][:n]) / n
pred = {L: 1 + c[:L].sum() for L in (1, 2, 3, 4, 256, 1024, 4096)}
sim = {1: 2.000, 2: 2.368, 3: 2.640, 4: 2.867, 256: 14.087, 1024: 26.954, 4096: 52.965}   # E30, exp_minus_1
se = {1: 0.0, 2: 0.001, 3: 0.002, 4: 0.002, 256: 0.021, 1024: 0.096, 4096: 0.388}
print("(b) Exp(1)-1:", " | ".join(f"L={L}: SA {pred[L]:.3f} sim {sim[L]:.3f}±{se[L]:.3f}" for L in pred))
out["b"] = dict(pred={str(k): v for k, v in pred.items()}, sim={str(k): v for k, v in sim.items()})

# (c) beta < 1: direct record counts vs persistence probabilities
rng = np.random.default_rng(1)
th = 1.0
res = []
for beta, mu, sig in ((0.9, 0.15, 0.3), (0.99, 0.0, 0.3), (0.99, 0.02, 0.3)):
    L, n = 2048, 4000
    I = mu + sig * rng.standard_normal((n, L))
    b = np.zeros((n, L)); acc = np.zeros(n)
    for k in range(L):
        acc = beta * acc + I[:, k]; b[:, k] = acc
    with np.errstate(over="ignore"):
        ak = (th - b) / beta ** np.arange(1, L + 1)
    run = np.minimum.accumulate(ak, axis=1)
    prev = np.concatenate([np.full((n, 1), np.inf), run[:, :-1]], axis=1)
    R = (ak < prev).sum(1)
    # independent estimate of pi_n from fresh inputs
    J = mu + sig * rng.standard_normal((n, L))
    W = np.cumsum(J * beta ** np.arange(L), axis=1)
    ok = W > th * (1 - beta ** np.arange(1, L + 1))
    pers = np.cumprod(ok, axis=1).mean(0)                         # pi_1 .. pi_L
    pi = np.concatenate([[1.0], pers])                             # pi_0 = 1
    pred_R = pi[:L].sum()
    tail = pers[-1]
    res.append(dict(beta=beta, mu=mu, L=L, records=float(R.mean()), records_se=float(R.std() / math.sqrt(n)), from_persistence=float(pred_R), pi_inf_est=float(tail)))
    print(f"(c) beta={beta} mu={mu}: E[R_L] direct {R.mean():.1f}±{R.std() / math.sqrt(n):.1f} | sum of persistence probs {pred_R:.1f} | pi_inf ~ {tail:.4f} (rate x L = {tail * L:.1f})")
out["c"] = res
json.dump(out, open("results/e30b_persistence.json", "w"), indent=1)
