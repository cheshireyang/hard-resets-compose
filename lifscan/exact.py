"""Exact (rational-arithmetic) reference implementation.

Everything here is written against a generic ordered field, so passing
``fractions.Fraction`` values gives bit-for-bit exact results.  This module is
the ground truth for the theory in docs/theory.md; the float/torch
implementations in ``lifscan.fast`` are checked against it.

Discrete hard-reset LIF (charge, then fire, then reset):

    v_t = beta * u_{t-1} + I_t
    s_t = [v_t >= theta]
    u_t = u_r if s_t else v_t

with 0 < beta <= 1.  As a map of the previous state u = u_{t-1}:

    f_t(u) = beta*u + I_t   if u <  c_t := (theta - I_t)/beta
             u_r            if u >= c_t
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

INF = None  # sentinel for +infinity breakpoint (no firing possible)


# ----------------------------------------------------------------------------
# serial reference
# ----------------------------------------------------------------------------

def serial_hard(I: Sequence, beta, theta, u_r, u0):
    """Return (u_1..u_T, s_1..s_T) by plain serial simulation."""
    u = u0
    us, ss = [], []
    for x in I:
        v = beta * u + x
        if v >= theta:
            u, s = u_r, 1
        else:
            u, s = v, 0
        us.append(u)
        ss.append(s)
    return us, ss


def serial_soft(I: Sequence, beta, theta, u0):
    """Subtractive (soft) reset, one subtraction per step."""
    u = u0
    us, ss = [], []
    for x in I:
        v = beta * u + x
        s = 1 if v >= theta else 0
        u = v - theta * s
        us.append(u)
        ss.append(s)
    return us, ss


# ----------------------------------------------------------------------------
# the hard-reset map family  H = { affine-then-steps }
# ----------------------------------------------------------------------------

@dataclass
class HardMap:
    """M(u) = A*u + B           for u < x[0]
              c[i]              for x[i] <= u < x[i+1]   (x[n] = +inf)

    ``x`` strictly increasing; ``len(x) == len(c)``.  An empty ``x`` means the
    segment can never fire (pure affine map).  A > 0 always.
    ``L`` = number of time steps the map covers.
    """

    A: object
    B: object
    x: List = field(default_factory=list)
    c: List = field(default_factory=list)
    L: int = 1

    # -- construction -------------------------------------------------------
    @staticmethod
    def step(I_t, beta, theta, u_r) -> "HardMap":
        return HardMap(beta, I_t, [(theta - I_t) / beta], [u_r], 1)

    @staticmethod
    def identity(one=1, zero=0) -> "HardMap":
        return HardMap(one, zero, [], [], 0)

    # -- evaluation -----------------------------------------------------------
    def __call__(self, u):
        x = self.x
        if not x or u < x[0]:
            return self.A * u + self.B
        # binary search: largest i with x[i] <= u
        lo, hi = 0, len(x) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if x[mid] <= u:
                lo = mid
            else:
                hi = mid - 1
        return self.c[lo]

    @property
    def pieces(self) -> int:
        return 1 + len(self.x)

    # -- composition:  (g after self)  -------------------------------------------
    def then(self, g: "HardMap", merge: bool = True) -> "HardMap":
        """Return g o self (apply self first, then g)."""
        F = self
        A, B = F.A, F.B
        nx: List = []
        nc: List = []
        # 1. H is affine on u < min(F.x[0], (g.x[0]-B)/A)  -- implicit in nx[0]
        # 2. g's step pieces pulled back into F's affine region (-inf, F.x[0])
        if g.x:
            for j, yj in enumerate(g.x):
                zj = (yj - B) / A
                if F.x and zj >= F.x[0]:
                    # this and all later g breakpoints fall outside F's affine region;
                    # but g's piece j-1 may still extend up to F.x[0] -- already open.
                    break
                nx.append(zj)
                nc.append(g.c[j])
        # 3. F's constant pieces, pushed through g
        for xi, ci in zip(F.x, F.c):
            nx.append(xi)
            nc.append(g(ci))
        if merge:
            nx, nc = _merge(nx, nc)
        return HardMap(g.A * A, g.A * B + g.B, nx, nc, F.L + g.L)


def _merge(x, c):
    """Drop breakpoints between two adjacent equal constants."""
    if not x:
        return x, c
    nx, nc = [x[0]], [c[0]]
    for xi, ci in zip(x[1:], c[1:]):
        if ci == nc[-1]:
            continue
        nx.append(xi)
        nc.append(ci)
    return nx, nc


def segment_map(I: Sequence, beta, theta, u_r, merge=True) -> HardMap:
    M = HardMap.step(I[0], beta, theta, u_r)
    for x in I[1:]:
        M = M.then(HardMap.step(x, beta, theta, u_r), merge=merge)
    return M


def closed_form_segment(I: Sequence, beta, theta, u_r):
    """Theorem 1 closed form: breakpoints are the strict prefix minima of
    a_k = (theta - b_k)/beta^k, constants are G_k (state after first firing at k).
    Returns (A, B, x, c) with x increasing (unmerged)."""
    L = len(I)
    b = 0
    a = []
    bk = []
    for k, x in enumerate(I, start=1):
        b = beta * b + x
        bk.append(b)
        a.append((theta - b) / beta ** k)
    # records: k is a record iff a_k < min_{j<k} a_j
    recs = []
    cur = None
    for k in range(1, L + 1):
        if cur is None or a[k - 1] < cur:
            recs.append(k)
            cur = a[k - 1]
    # G_k = state at L after reset at k, i.e. serial from u_r over I[k:]
    xs, cs = [], []
    for k in reversed(recs):  # largest k has smallest a_k -> leftmost breakpoint
        xs.append(a[k - 1])
        tail, _ = serial_hard(I[k:], beta, theta, u_r, u_r)
        cs.append(tail[-1] if tail else u_r)
    return beta ** L, bk[-1], xs, cs


# ----------------------------------------------------------------------------
# Blelloch-style tree scan (exact), returns all states
# ----------------------------------------------------------------------------

def tree_scan_states(I: Sequence, beta, theta, u_r, u0, merge=True,
                     stats: Optional[dict] = None):
    """Up-sweep builds segment maps over a balanced binary tree; down-sweep
    pushes *scalar* states down (never materialises prefix maps).
    Returns (u_1..u_T, s_1..s_T)."""
    T = len(I)
    leaves = [HardMap.step(x, beta, theta, u_r) for x in I]

    def build(lo, hi):
        if hi - lo == 1:
            return (leaves[lo], None, None, lo, hi)
        mid = (lo + hi) // 2
        Lc, Rc = build(lo, mid), build(mid, hi)
        M = Lc[0].then(Rc[0], merge=merge)
        if stats is not None:
            stats.setdefault("max_pieces", 0)
            stats["max_pieces"] = max(stats["max_pieces"], M.pieces)
            stats["total_pieces"] = stats.get("total_pieces", 0) + M.pieces
        return (M, Lc, Rc, lo, hi)

    root = build(0, T)
    start = [None] * T  # state entering step t (= u_{t-1})

    def down(node, u_in):
        M, Lc, Rc, lo, hi = node
        if Lc is None:
            start[lo] = u_in
            return
        down(Lc, u_in)
        down(Rc, Lc[0](u_in))

    down(root, u0)
    us, ss = [], []
    for t in range(T):
        v = beta * start[t] + I[t]
        if v >= theta:
            us.append(u_r)
            ss.append(1)
        else:
            us.append(v)
            ss.append(0)
    return us, ss


# ----------------------------------------------------------------------------
# generic piecewise-affine maps (used to count pieces for soft reset fairly)
# ----------------------------------------------------------------------------

@dataclass
class PWA:
    """Piecewise affine map on R.  Pieces [x_i, x_{i+1}) with x_0 = -inf.
    ``bps`` = interior breakpoints (len n-1), ``ab`` = [(slope, intercept)]*n.
    All slopes are > 0 for the LIF families considered."""

    bps: List
    ab: List[Tuple]

    def __call__(self, u):
        i = 0
        while i < len(self.bps) and self.bps[i] <= u:
            i += 1
        a, b = self.ab[i]
        return a * u + b

    @property
    def pieces(self):
        return len(self.ab)

    def then(self, g: "PWA") -> "PWA":
        nb, nab = [], []
        n = len(self.ab)
        for i, (a, b) in enumerate(self.ab):
            lo = self.bps[i - 1] if i > 0 else None
            hi = self.bps[i] if i < n - 1 else None
            if a < 0:
                raise ValueError("PWA.then assumes non-negative slopes; use count_pieces_affine for order-reversing resets")
            # image interval of this piece is (a*lo+b, a*hi+b) (a>0)
            # split by g breakpoints lying strictly inside the image
            if i > 0:
                nb.append(lo)
            if a == 0:  # constant piece: image is the single point b
                j = 0
                while j < len(g.bps) and g.bps[j] <= b:
                    j += 1
                ga, gb = g.ab[j]
                nab.append((0 * ga, ga * b + gb))
                continue
            # first g piece index active at the left end
            if lo is None:
                j = 0
            else:
                ylo = a * lo + b
                j = 0
                while j < len(g.bps) and g.bps[j] <= ylo:
                    j += 1
            while True:
                ga, gb = g.ab[j]
                nab.append((ga * a, ga * b + gb))
                if j < len(g.bps):
                    yb = g.bps[j]
                    zb = (yb - b) / a
                    if hi is None or zb < hi:
                        nb.append(zb)
                        j += 1
                        continue
                break
        return _pwa_merge(PWA(nb, nab))


def _pwa_merge(F: PWA) -> PWA:
    """Merge adjacent pieces with identical affine maps (continuous joins)."""
    if not F.bps:
        return F
    nb, nab = [], [F.ab[0]]
    for bp, ab in zip(F.bps, F.ab[1:]):
        if ab == nab[-1]:
            continue
        nb.append(bp)
        nab.append(ab)
    return PWA(nb, nab)


def pwa_step_hard(I_t, beta, theta, u_r):
    # constant piece encoded as slope 0 -- allowed here
    return PWA([(theta - I_t) / beta], [(beta, I_t), (0 * beta, u_r)])


def pwa_step_soft(I_t, beta, theta):
    return PWA([(theta - I_t) / beta], [(beta, I_t), (beta, I_t - theta)])


def pwa_segment(steps: List[PWA]) -> PWA:
    M = steps[0]
    for s in steps[1:]:
        M = M.then(s)
    return M


def count_pieces_affine(I: Sequence, beta, theta, gamma, delta, lo=None, hi=None, return_pieces=False):
    """Number of pieces (maximal intervals of initial states with a constant spike pattern) of the L-step map of
    v = beta*u + I_t;  u <- gamma*v + delta if v >= theta else v,
    on the interval (lo, hi) (None = unbounded).  Exact for exact inputs; valid for any sign of beta*gamma,
    unlike ``PWA.then``.  Each piece is tracked as (x_lo, x_hi, a, b, pattern) with u_t = a*x + b."""
    P = [(lo, hi, 1 + 0 * beta, 0 * beta, ())]
    for It in I:
        Q = []
        for (xl, xr, a, b, pat) in P:
            va, vb = beta * a, beta * b + It

            def emit(l, r, spike):
                if spike:
                    Q.append((l, r, gamma * va, gamma * vb + delta, pat + (1,)))
                else:
                    Q.append((l, r, va, vb, pat + (0,)))

            if va == 0:
                emit(xl, xr, vb >= theta)
                continue
            z = (theta - vb) / va
            if (xl is None or xl < z) and (xr is None or z < xr):
                if va > 0:
                    emit(xl, z, False); emit(z, xr, True)
                else:
                    emit(xl, z, True); emit(z, xr, False)
            else:
                # the whole piece lies on one side of the threshold: test an interior point
                if xl is not None and xr is not None:
                    x = (xl + xr) / 2
                elif xl is not None:
                    x = xl + 1
                else:
                    x = xr - 1
                emit(xl, xr, va * x + vb >= theta)
        P = Q
    n = 1 + sum(1 for i in range(1, len(P)) if P[i][4] != P[i - 1][4])
    return (n, P) if return_pieces else n

