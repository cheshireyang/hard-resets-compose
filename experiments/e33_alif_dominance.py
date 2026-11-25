"""E33: ALIF spike-time dominance (Theorem 9) -- exact (rational) checks.

Model: v_t = beta*u_{t-1} + I_t ; s_t = [v_t >= theta_t + w_{t-1}] ; u_t = u_r if s_t else v_t ; w_t = rho*w_{t-1} + q*s_t.

Theorem.  Let beta > 0, 0 < rho <= 1, q >= 0, and assume every post-reset subthreshold state is >= u_r
(sufficient: I_t >= (1-beta)*u_r for all t).  If u_0 >= u_0' and w_0 < w_0', then the i-th spike of the first
trajectory occurs no later than the i-th spike of the second, for every i.  Consequences: N_t >= N_t', and
w_t < w_t' whenever N_t = N_t'  (the monotonicity conjecture of App. ALIF, which is false without the
assumptions), and the post-reset map Psi on any w-interval has at most 1 + L(L+1)/2 pieces.

This script
 (1) tests the theorem on random exact instances (incl. general entry states and time-varying thresholds),
 (2) verifies explicit counterexamples when either assumption is dropped and measures violation rates,
 (3) verifies witnesses that attain the piece bound for L <= 5 and records local-search maxima for larger L,
 (4) counts, over the fitted Allen ALIF models (E28all), the stimulus level above which the sufficient
     condition I_t >= (1-beta) u_r holds.
Output: results/e33_alif_dominance.json
"""
import glob, json, os, random, sys
from fractions import Fraction as Fr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run(I, u, w, beta, rho, q, ur, theta):
    """Return spike times (1-based) and the (N, w, u) history."""
    times, hist = [], []
    for t, x in enumerate(I, 1):
        v = beta * u + x
        th = theta[t - 1] if isinstance(theta, list) else theta
        if v >= th + w:
            u = ur; w = rho * w + q; times.append(t)
        else:
            u = v; w = rho * w
        hist.append((len(times), w, u))
    return times, hist


def dominance_ok(ta, tb):
    """a_i <= b_i for every spike i of the second trajectory (absent spikes = +inf)."""
    return len(ta) >= len(tb) and all(a <= b for a, b in zip(ta, tb))


def conj_ok(ha, hb):
    """Conjecture: N >= N', and w < w' when N = N', at every step."""
    for (na, wa, _), (nb, wb, _) in zip(ha, hb):
        if na < nb or (na == nb and not wa < wb):
            return False
    return True


def psi_pieces(I, beta, rho, q, theta, ur, wlo, whi):
    """Exact pieces of the post-reset map w -> (u_L, w_L) on [wlo, whi): intervals with constant (slope, offset, u)."""
    iv = [(wlo, whi, Fr(1), Fr(0), ur)]
    for x in I:
        new = []
        for lo, hi, a, b, u in iv:
            v = beta * u + x
            z = (v - theta - b) / a          # spike iff w <= z
            if z >= lo:
                new.append((lo, min(hi, z), rho * a, rho * b + q, ur))
            if z < hi:
                new.append((max(lo, z), hi, rho * a, rho * b, v))
        new.sort(key=lambda r: (r[0], r[1]))
        merged = []
        for r in new:
            if r[1] < r[0]:
                continue
            if merged and merged[-1][2:] == r[2:] and merged[-1][1] >= r[0]:
                m = merged[-1]; merged[-1] = (m[0], max(m[1], r[1])) + m[2:]
            else:
                merged.append(r)
        iv = [r for r in merged if r[1] > r[0]]
    return len(iv)


def rnd_instance(rng, cond, rho_max=Fr(1)):
    L = rng.randint(2, 12)
    beta = Fr(rng.randint(1, 40), 20)                    # beta in (0, 2]
    rho = Fr(rng.randint(1, int(20 * rho_max)), 20)
    q = Fr(rng.randint(0, 20), 10)
    ur = Fr(rng.randint(-20, 9), 10)
    tv = rng.random() < 0.5                              # time-varying threshold
    theta = [Fr(rng.randint(5, 15), 10) for _ in range(L)] if tv else Fr(1)
    lo = (1 - beta) * ur if cond else Fr(-3)
    I = [lo + Fr(rng.randint(0, 40), 10) for _ in range(L)]
    w0 = Fr(rng.randint(0, 40), 10); w1 = w0 + Fr(rng.randint(1, 20), 20)
    if rng.random() < 0.5:                               # post-reset entry (Psi) or general entry with u0 >= u0'
        u0 = u1 = ur
    else:
        u1 = Fr(rng.randint(-20, 10), 10); u0 = u1 + Fr(rng.randint(0, 10), 10)
    return I, (u0, w0), (u1, w1), (beta, rho, q, ur, theta)


def main():
    rng = random.Random(0)
    out = {}
    # (1) theorem under its assumptions
    n_tests = int(os.environ.get("E33_N", 200000))
    viol = 0
    for _ in range(n_tests):
        I, (u0, w0), (u1, w1), p = rnd_instance(rng, cond=True)
        ta, ha = run(I, u0, w0, *p); tb, hb = run(I, u1, w1, *p)
        if not (dominance_ok(ta, tb) and conj_ok(ha, hb)):
            viol += 1
    out["theorem"] = dict(tests=n_tests, violations=viol)
    print("theorem:", out["theorem"], flush=True)

    # (2) assumptions dropped: random violation rates of the conjecture
    rates = {}
    for name, cond, rho_max in (("upward_reset_allowed", False, Fr(1)), ("rho_up_to_2", True, Fr(2))):
        v = 0
        for _ in range(n_tests):
            I, (u0, w0), (u1, w1), p = rnd_instance(rng, cond=cond, rho_max=rho_max)
            ta, ha = run(I, u0, w0, *p); tb, hb = run(I, u1, w1, *p)
            v += not conj_ok(ha, hb)
        rates[name] = dict(tests=n_tests, violations=v)
        print(name, rates[name], flush=True)
    out["dropped"] = rates

    # explicit counterexamples (verified exactly)
    ce = []
    # (a) upward reset, rho <= 1, q = 0 and q = 1/100: the earlier spiker is overtaken in spike count
    for q in (Fr(0), Fr(1, 100)):
        I = [Fr(19, 10), Fr(-19, 10), Fr(13, 10), Fr(9, 10)]
        p = (Fr(3, 4), Fr(1, 20), q, Fr(3, 10), Fr(1))
        ta, ha = run(I, p[3], Fr(9, 10), *p); tb, hb = run(I, p[3], Fr(27, 20), *p)
        ce.append(dict(kind="upward reset", beta="3/4", rho="1/20", q=str(q), ur="3/10", I=[str(x) for x in I],
                       w=["9/10", "27/20"], spikes=[ta, tb], conj_holds=conj_ok(ha, hb),
                       cond_holds=all(x >= (1 - p[0]) * p[3] for x in I)))
    # (b) rho > 1 with downward reset: equal counts but w_t >= w_t'
    I = [Fr(29, 10), Fr(2)]
    p = (Fr(9, 10), Fr(7, 5), Fr(11, 10), Fr(-2), Fr(1))
    ta, ha = run(I, p[3], Fr(1, 10), *p); tb, hb = run(I, p[3], Fr(1, 5), *p)
    ce.append(dict(kind="rho > 1", beta="9/10", rho="7/5", q="11/10", ur="-2", I=[str(x) for x in I], w=["1/10", "1/5"],
                   spikes=[ta, tb], final=[[str(ha[-1][0]), str(ha[-1][1])], [str(hb[-1][0]), str(hb[-1][1])]],
                   conj_holds=conj_ok(ha, hb), cond_holds=all(x >= (1 - p[0]) * p[3] for x in I)))
    for c in ce:
        assert not c["conj_holds"] and (c["cond_holds"] == (c["kind"] == "rho > 1")), c
        print("counterexample:", c, flush=True)
    out["counterexamples"] = ce

    # (3) piece bound 1 + L(L+1)/2 on Psi: witnesses for L <= 5, local search for L <= 8
    wit = [(3, Fr(17, 20), Fr(39, 40), Fr(7, 4), Fr(3, 4), [Fr(357, 25), Fr(339, 50), Fr(433, 100)]),
           (4, Fr(3, 40), Fr(33, 40), Fr(41, 20), Fr(-2, 5), [Fr(182, 25), Fr(289, 50), Fr(623, 100), Fr(683, 100)]),
           (5, Fr(1, 4), Fr(33, 40), Fr(9, 5), Fr(1, 4), [Fr(5, 2), Fr(39, 10), Fr(24, 5), Fr(59, 10), Fr(131, 20)])]
    witnesses = []
    for L, beta, rho, q, ur, I in wit:
        assert all(x >= (1 - beta) * ur for x in I)
        k = psi_pieces(I, beta, rho, q, Fr(1), ur, Fr(0), Fr(100))
        witnesses.append(dict(L=L, pieces=k, bound=1 + L * (L + 1) // 2, beta=str(beta), rho=str(rho), q=str(q),
                              ur=str(ur), I=[str(x) for x in I]))
        print("witness", witnesses[-1], flush=True)
        assert k == 1 + L * (L + 1) // 2
    out["witnesses"] = witnesses

    search = []
    for L in range(2, 9):
        best = 0
        for restart in range(int(os.environ.get("E33_RESTARTS", 12))):
            beta = Fr(rng.randint(10, 20), 20); rho = Fr(rng.randint(10, 20), 20)
            q = Fr(rng.randint(1, 30), 10); ur = Fr(rng.randint(-10, 5), 10)
            I = [(1 - beta) * ur + Fr(rng.randint(0, 30), 10) for _ in range(L)]
            cur = psi_pieces(I, beta, rho, q, Fr(1), ur, Fr(0), Fr(100))
            for it in range(int(os.environ.get("E33_ITERS", 1500))):
                I2, b2, r2, q2, u2 = list(I), beta, rho, q, ur
                k = rng.randrange(L + 2)
                if k < L:
                    I2[k] += Fr(rng.randint(-10, 10), 20)
                elif k == L:
                    b2 = min(Fr(1), max(Fr(1, 20), beta + Fr(rng.randint(-2, 2), 40)))
                    r2 = min(Fr(1), max(Fr(1, 20), rho + Fr(rng.randint(-2, 2), 40)))
                else:
                    q2 = max(Fr(0), q + Fr(rng.randint(-3, 3), 20)); u2 = min(Fr(9, 10), ur + Fr(rng.randint(-2, 2), 20))
                if any(x < (1 - b2) * u2 for x in I2):
                    continue
                c = psi_pieces(I2, b2, r2, q2, Fr(1), u2, Fr(0), Fr(100))
                if c >= cur:
                    I, beta, rho, q, ur, cur = I2, b2, r2, q2, u2, c
            best = max(best, cur)
        search.append(dict(L=L, max_found=best, bound=1 + L * (L + 1) // 2))
        print("search", search[-1], flush=True)
    out["search"] = search

    # (4) Allen ALIF fits: stimulus level (pA) above which I_t >= (1-beta) u_r
    th = []; n_pos_ur = 0
    for f in sorted(glob.glob("results/e28all_allen_alif_w*.json")):
        for r in json.load(open(f)):
            if "beta" in r:
                n_pos_ur += r["ur"] >= 0
                th.append(((1 - r["beta"]) * r["ur"] - r["bias"]) / r["gain"] * 100.0)   # X is in 100 pA units
    th.sort()
    if th:
        out["allen"] = dict(cells=len(th), median_pA=th[len(th) // 2], q95_pA=th[int(0.95 * (len(th) - 1))],
                            max_pA=th[-1], cells_needing_positive_stimulus=sum(t > 0 for t in th), cells_positive_ur=n_pos_ur)
        print("allen", out["allen"], flush=True)
    json.dump(out, open("results/e33_alif_dominance.json", "w"), indent=1)


if __name__ == "__main__":
    main()
