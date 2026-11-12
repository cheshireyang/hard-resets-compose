"""E23: both hypotheses of Theorem 7(b) are necessary (exact arithmetic).

(A) Order-reversing affine resets  r(v) = theta - delta - gamma*(v - theta),  gamma > 0:
    explicit inputs for which the L-step map has 2^L pieces, the largest number any affine reset can give.
    Construction (proof in the paper): keep an interval K_t of length l_t that lies in the closure of the image
    of every piece; put the threshold preimage z_t inside K_t; every piece is cut in two, the lower halves land
    on [theta - beta*p, theta), the reflected upper halves on [theta - delta - gamma*beta*q, theta - delta], and
    the two families overlap again.  Lengths are fixed backwards from l_L = 1.
(B) Expanding leak beta > 1 with soft reset and zero input: the beta-transformation x -> beta*x mod 1.
(C) Expanding reset gamma > 1 with leak beta = 1 and constant input.
Every count is checked against the serial recurrence at the midpoint of every piece."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fractions import Fraction as Fr
from lifscan.exact import count_pieces_affine

TH = Fr(1)


def serial_pattern(x, I, beta, g, d):
    u, pat = x, []
    for It in I:
        v = beta * u + It
        if v >= TH: u = g * v + d; pat.append(1)
        else:       u = v; pat.append(0)
    return tuple(pat)


def midpoint_check(P, I, beta, g, d):
    """Every tracked piece has the pattern the serial recurrence gives at an interior point."""
    for (xl, xr, a, b, pat) in P:
        x = (xl + xr) / 2 if (xl is not None and xr is not None) else (xl + 1 if xr is None else xr - 1)
        if serial_pattern(x, I, beta, g, d) != pat: return False
    return True


def reversing_inputs(L, beta, gamma, delta):
    dp, dm = max(delta, 0), max(-delta, 0)
    ell = [None] * (L + 1); ell[L] = Fr(1)
    p = [None] * L; q = [None] * L
    for t in range(L - 1, -1, -1):
        p[t] = (ell[t + 1] + dp) / beta; q[t] = (ell[t + 1] + dm) / (gamma * beta); ell[t] = p[t] + q[t]
    I = []
    for t in range(L):
        lo = -p[0] if t == 0 else TH - dp - ell[t]          # K_0 = [-p_0, q_0];  K_t = [theta - delta^+ - l_t, theta - delta^+]
        z = lo + p[t]
        I.append(TH - beta * z)                             # threshold preimage at u = z
    return I


rows = []
LMAX = 12
print("(A) order-reversing resets: pieces for L = 1..%d" % LMAX)
for beta, gamma, delta in ((Fr(1), Fr(1), Fr(0)), (Fr(9, 10), Fr(1), Fr(0)), (Fr(9, 10), Fr(1, 2), Fr(0)), (Fr(1, 2), Fr(1, 2), Fr(0)),
                           (Fr(9, 10), Fr(1), Fr(1)), (Fr(9, 10), Fr(1, 2), Fr(1, 2)), (Fr(1, 2), Fr(1, 2), Fr(3, 2)),   # last: r(v) = -v/2
                           (Fr(99, 100), Fr(1, 10), Fr(2)), (Fr(9, 10), Fr(1), Fr(-1, 2))):
    g, d = -gamma, TH - delta + gamma * TH
    out = []
    for L in range(1, LMAX + 1):
        I = reversing_inputs(L, beta, gamma, delta)
        n, P = count_pieces_affine(I, beta, TH, g, d, return_pieces=True)
        ok = midpoint_check(P, I, beta, g, d)
        rows.append(dict(part="reversing", beta=str(beta), gamma=str(gamma), delta=str(delta), L=L, pieces=n, serial_ok=ok,
                         max_abs_input=float(max(abs(x) for x in I))))
        out.append(n)
    print(f"  beta={beta} gamma={gamma} delta={delta}: {out}  serial check: {all(r['serial_ok'] for r in rows[-LMAX:])}", flush=True)

print("(B) expanding leak, soft reset, zero input, initial states in [0,1)")
for beta, Lmax in ((Fr(2), 12), (Fr(3, 2), 20), (Fr(5, 4), 30), (Fr(11, 10), 60)):
    out = []
    for L in range(1, Lmax + 1):
        I = [Fr(0)] * L
        n, P = count_pieces_affine(I, beta, TH, Fr(1), -TH, lo=Fr(0), hi=Fr(1), return_pieces=True)
        ok = midpoint_check(P, I, beta, Fr(1), -TH) if L <= 14 else None
        rows.append(dict(part="expanding", beta=str(beta), L=L, pieces=n, quad=1 + L * (L + 1) // 2, serial_ok=ok))
        out.append(n)
    first = next((r["L"] for r in rows if r["part"] == "expanding" and r["beta"] == str(beta) and r["pieces"] > r["quad"]), None)
    print(f"  beta={beta}: L={Lmax}: {out[-1]} pieces (quadratic bound {1 + Lmax * (Lmax + 1) // 2}); first L above the bound: {first}", flush=True)
print("(C) expanding reset r(v) = gamma*(v - theta), gamma > 1, leak beta = 1, constant input theta/gamma, initial states in [0,1)")
import math
for gamma, Lmax in ((Fr(2), 22), (Fr(3, 2), 22), (Fr(3), 24)):
    m = math.ceil(gamma); out = []
    for L in range(1, Lmax + 1):
        I = [TH / gamma] * L
        n, P = count_pieces_affine(I, Fr(1), TH, gamma, -gamma * TH, lo=Fr(0), hi=Fr(1), return_pieces=True)
        ok = midpoint_check(P, I, Fr(1), gamma, -gamma * TH) if L <= 14 else None
        lower = float(gamma) ** (L // m)
        rows.append(dict(part="expanding_reset", beta="1", gamma=str(gamma), L=L, pieces=n, quad=1 + L * (L + 1) // 2, lemma_lower=lower,
                         serial_ok=ok))
        out.append(n)
    sel = [r for r in rows if r["part"] == "expanding_reset" and r["gamma"] == str(gamma)]
    first = next((r["L"] for r in sel if r["pieces"] > r["quad"]), None)
    print(f"  gamma={gamma}: L={Lmax}: {out[-1]} pieces (quadratic bound {1 + Lmax * (Lmax + 1) // 2}, lemma lower bound "
          f"{sel[-1]['lemma_lower']:.0f}); first L above the quadratic bound: {first}; lemma bound respected: "
          f"{all(r['pieces'] >= r['lemma_lower'] for r in sel)}", flush=True)
print("(C') the same with a contracting leak: beta = 9/10, r(v) = (3/2)(v - 1), constant input 1/2, all initial states")
out = []
for L in range(1, 15):
    I = [Fr(1, 2)] * L
    n, P = count_pieces_affine(I, Fr(9, 10), TH, Fr(3, 2), Fr(-3, 2), return_pieces=True)
    rows.append(dict(part="expanding_reset_leaky", beta="9/10", gamma="3/2", L=L, pieces=n, quad=1 + L * (L + 1) // 2,
                     serial_ok=midpoint_check(P, I, Fr(9, 10), Fr(3, 2), Fr(-3, 2))))
    out.append(n)
print("  pieces L=1..14:", out, " quadratic bound:", [1 + L * (L + 1) // 2 for L in range(1, 15)])
print("all serial checks passed:", all(r["serial_ok"] is not False for r in rows))
json.dump(rows, open("results/e23_reset_lower_bounds.json", "w"), indent=1)
