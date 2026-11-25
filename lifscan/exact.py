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
# O(n_F + n_g) linear-time composition and O(T log T) exact tree scan
# ----------------------------------------------------------------------------

@dataclass
class LinearHardMap:
    """Hard-reset segment map with sorted distinct output constants ``v`` and
    per-piece indices ``cid``, supporting exact O(n_F + n_g) composition.

    M(u) = A*u + B          for u < x[0]
           v[cid[i]]        for x[i] <= u < x[i+1]   (x[n] = +inf)

    Invariants: ``x`` strictly increasing, ``v`` strictly increasing,
    ``len(x) == len(cid)``, every entry of ``v`` is referenced by ``cid``
    (hence ``len(v) <= len(x)``), and ``A > 0``.
    """

    A: object
    B: object
    x: List = field(default_factory=list)
    v: List = field(default_factory=list)
    cid: List[int] = field(default_factory=list)
    L: int = 1

    @staticmethod
    def step(I_t, beta, theta, u_r) -> "LinearHardMap":
        return LinearHardMap(beta, I_t, [(theta - I_t) / beta], [u_r], [0], 1)

    @staticmethod
    def identity(one=1, zero=0) -> "LinearHardMap":
        return LinearHardMap(one, zero, [], [], [], 0)

    @property
    def pieces(self) -> int:
        return 1 + len(self.x)

    @property
    def c(self) -> List:
        return [self.v[i] for i in self.cid]

    def __call__(self, u):
        x = self.x
        if not x or u < x[0]:
            return self.A * u + self.B
        lo, hi = 0, len(x) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if x[mid] <= u:
                lo = mid
            else:
                hi = mid - 1
        return self.v[self.cid[lo]]

    def then(self, g: "LinearHardMap", merge: bool = True,
             op_counter: Optional[List[int]] = None) -> "LinearHardMap":
        """Exact O(n_F + n_g) composition ``g o self`` using two linear merges."""
        F = self
        A, B = F.A, F.B
        Ap, Bp = g.A, g.B
        p, q = len(F.x), len(F.v)
        r, s = len(g.x), len(g.v)
        ops = 0

        if r == 0:
            nv = [Ap * u + Bp for u in F.v]
            ops += q
            if op_counter is not None:
                op_counter[0] += ops
            return LinearHardMap(Ap * A, Ap * B + Bp, list(F.x), nv, list(F.cid), F.L + g.L)

        # 1. Linear two-pointer evaluation of g on F's sorted values F.v
        a_vals = []
        u_to_gvid = [0] * q
        y0 = g.x[0]
        k = 0
        while k < q and F.v[k] < y0:
            a_vals.append(Ap * F.v[k] + Bp)
            k += 1
            ops += 1
        j = 0
        for t in range(k, q):
            ut = F.v[t]
            while j + 1 < r and g.x[j + 1] <= ut:
                j += 1
                ops += 1
            u_to_gvid[t] = g.cid[j]
            ops += 1

        # 2. Concatenate pulled-back breakpoints of g (< F.x[0]) with F's breakpoints
        raw_x = []
        raw_is_w = []
        raw_idx = []
        used_a = [False] * k
        used_w = [False] * s

        x0 = F.x[0] if p > 0 else None
        for j in range(r):
            zj = (g.x[j] - B) / A
            ops += 1
            if x0 is not None and zj >= x0:
                break
            wid = g.cid[j]
            raw_x.append(zj)
            raw_is_w.append(True)
            raw_idx.append(wid)
            used_w[wid] = True

        for i in range(p):
            t = F.cid[i]
            raw_x.append(F.x[i])
            if t < k:
                raw_is_w.append(False)
                raw_idx.append(t)
                used_a[t] = True
            else:
                wid = u_to_gvid[t]
                raw_is_w.append(True)
                raw_idx.append(wid)
                used_w[wid] = True
            ops += 1

        # 3. Linear two-way merge of used affine images (sorted) and used g.v (sorted)
        a_list = [t for t in range(k) if used_a[t]]
        w_list = [b for b in range(s) if used_w[b]]
        rank_a = [-1] * k
        rank_w = [-1] * s
        nv = []
        ia, iw = 0, 0
        na, nw = len(a_list), len(w_list)
        while ia < na and iw < nw:
            ta, wb = a_list[ia], w_list[iw]
            va, vw = a_vals[ta], g.v[wb]
            ops += 1
            if va < vw:
                rank_a[ta] = len(nv)
                nv.append(va)
                ia += 1
            elif vw < va:
                rank_w[wb] = len(nv)
                nv.append(vw)
                iw += 1
            else:
                idx_new = len(nv)
                rank_a[ta] = idx_new
                rank_w[wb] = idx_new
                nv.append(va)
                ia += 1
                iw += 1
        while ia < na:
            ta = a_list[ia]
            rank_a[ta] = len(nv)
            nv.append(a_vals[ta])
            ia += 1
            ops += 1
        while iw < nw:
            wb = w_list[iw]
            rank_w[wb] = len(nv)
            nv.append(g.v[wb])
            iw += 1
            ops += 1

        # 4. Remap piece value indices and drop adjacent equal pieces
        nx = []
        ncid = []
        for xi, is_w, idx in zip(raw_x, raw_is_w, raw_idx):
            c_new = rank_w[idx] if is_w else rank_a[idx]
            ops += 1
            if merge and ncid and ncid[-1] == c_new:
                continue
            nx.append(xi)
            ncid.append(c_new)

        # Compact value table if adjacent deduplication dropped any value
        if merge and nv:
            still_used = [False] * len(nv)
            for c_new in ncid:
                still_used[c_new] = True
            if not all(still_used):
                remap = [-1] * len(nv)
                nv2 = []
                for idx_v, ok in enumerate(still_used):
                    if ok:
                        remap[idx_v] = len(nv2)
                        nv2.append(nv[idx_v])
                nv = nv2
                ncid = [remap[c] for c in ncid]
                ops += len(still_used) + len(ncid)

        if op_counter is not None:
            op_counter[0] += ops
        return LinearHardMap(Ap * A, Ap * B + Bp, nx, nv, ncid, F.L + g.L)


def tree_scan_states_linear(I: Sequence, beta, theta, u_r, u0, merge: bool = True,
                            stats: Optional[dict] = None):
    """Exact Blelloch-style tree scan with O(n_F + n_g) composition:
    O(T log T) total up-sweep work and O(T) down-sweep work."""
    T = len(I)
    leaves = [LinearHardMap.step(x, beta, theta, u_r) for x in I]
    ops = [0]

    def build(lo, hi):
        if hi - lo == 1:
            return (leaves[lo], None, None, lo, hi)
        mid = (lo + hi) // 2
        Lc, Rc = build(lo, mid), build(mid, hi)
        M = Lc[0].then(Rc[0], merge=merge, op_counter=ops)
        if stats is not None:
            stats.setdefault("max_pieces", 0)
            stats["max_pieces"] = max(stats["max_pieces"], M.pieces)
            stats["total_pieces"] = stats.get("total_pieces", 0) + M.pieces
        return (M, Lc, Rc, lo, hi)

    root = build(0, T)
    if stats is not None:
        stats["compose_ops"] = ops[0]
    start = [None] * T

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

