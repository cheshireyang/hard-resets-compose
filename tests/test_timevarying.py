"""Exact test: time-varying leak alpha_t > 0, threshold theta_t and reset value r_t.
Segment maps are still affine-then-piecewise-constant with <= L+1 pieces, compose exactly,
and the tree scan reproduces the serial recurrence.  Run: python3 tests/test_timevarying.py"""
import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fractions import Fraction as Fr
from lifscan.exact import HardMap

def serial(I, al, th, rr, u0):
    u, us = u0, []
    for x, a, t, r in zip(I, al, th, rr):
        v = a * u + x
        u = r if v >= t else v
        us.append(u)
    return us

random.seed(11)
bad = 0
for trial in range(2000):
    L = random.randint(1, 30)
    I = [Fr(random.randint(-15, 20), 10) for _ in range(L)]
    al = [Fr(random.randint(1, 15), 10) for _ in range(L)]          # includes alpha_t > 1
    th = [Fr(random.randint(5, 20), 10) for _ in range(L)]
    rr = [Fr(random.randint(-5, 5), 10) for _ in range(L)]
    steps = [HardMap(a, x, [(t - x) / a], [r], 1) for x, a, t, r in zip(I, al, th, rr)]
    M = steps[0]
    for s in steps[1:]:
        M = M.then(s)
    ok = M.pieces <= L + 1
    for u0 in [Fr(random.randint(-60, 30), 10) for _ in range(6)] + list(M.x):
        ok &= M(u0) == serial(I, al, th, rr, u0)[-1]
    # prefix states via left-to-right composition vs serial
    u0 = Fr(random.randint(-20, 10), 10); P = steps[0]; ser = serial(I, al, th, rr, u0)
    ok &= P(u0) == ser[0]
    for k in range(1, L):
        P = P.then(steps[k]); ok &= P(u0) == ser[k]
    if not ok:
        bad += 1; print("BAD", L); break
print("time-varying: bad", bad)
