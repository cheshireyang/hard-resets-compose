"""Exact two-level scan for adaptive-threshold (ALIF) neurons with a predicted adaptation box.

    v = beta*u + I_t ;  s = [v >= theta0 + w] ;  u <- u_r if s else v ;  w <- rho*w + q*s

A chunk map (u0, w0) -> (u_C, w_C) is built only on an entry box [lu, hu) x [lw, hw]:
  * first-spike structure: an unspiked state fires at step j iff u0 >= g_j(w0) = (theta0 + rho^(j-1) w0 - b_j)/beta^j,
    with b_j the zero-start free trajectory.  We keep the candidate lines g_j that can be the first crossing somewhere
    in the box (pruning: single-line dominance on the feasible w-window, the box edges, and the w-range already
    forced to spike by u0 >= lu);
  * after a first spike at step k the state is (u_r, x) with x = rho^k w0 + q, and the rest of the chunk is a
    1-D map of x (App. D of the paper): intervals of x with a constant final u and a final w affine in x.  They are
    tracked as explicit intervals and split at threshold crossings, with the prefix-sum compaction of the soft scan;
  * no spike: (beta^C u0 + b_C, rho^C w0).
Any box is exact: the carry checks that the entry state lies in it, and otherwise (or on capacity overflow)
evaluates the chunk serially.  The box in w is centred on a prediction (e.g. the previous optimisation iterate, or
a short warm-up); lu comes from the lower envelope of the hard-reset algebra, which also bounds ALIF potentials,
and hu = max(u_r, theta0 + hw/rho).  Replay and repair are as for the hard reset."""
import math
import torch
import triton
import triton.language as tl
from . import triton_lif as L


@triton.jit
def _alif_step(u, w, x, beta, rho, q, theta0, u_r):
    v = beta * u + x
    s = v >= theta0 + w
    u = tl.where(s, u_r, v)
    w = tl.where(s, rho * w + q, rho * w)
    return u, w, s


@triton.jit
def alif_serial_kernel(I, U0, W0, Uout, Wout, Sout, T, N, beta, rho, q, theta0, u_r, BN: tl.constexpr):
    pid = tl.program_id(0); n = pid * BN + tl.arange(0, BN); mask = n < N
    u = tl.load(U0 + n, mask=mask, other=0.0); w = tl.load(W0 + n, mask=mask, other=0.0)
    for t in range(T):
        x = tl.load(I + t * N + n, mask=mask, other=0.0)
        u, w, s = _alif_step(u, w, x, beta, rho, q, theta0, u_r)
        tl.store(Uout + t * N + n, u, mask=mask); tl.store(Wout + t * N + n, w, mask=mask)
        tl.store(Sout + t * N + n, s.to(tl.float32), mask=mask)


@triton.jit
def alif_tiled_kernel(I, SU, SW, Uout, Wout, Sout, EU, EW, N, Ls, beta, rho, q, theta0, u_r,
                      BN: tl.constexpr, D: tl.constexpr):
    """Time-tiled recurrence over segment m of length Ls (serial baseline: one segment; replay: chunks)."""
    m = tl.program_id(0); pid = tl.program_id(1)
    n = pid * BN + tl.arange(0, BN); mask = n < N
    rows = tl.arange(0, D)
    u = tl.load(SU + m * N + n, mask=mask, other=0.0); w = tl.load(SW + m * N + n, mask=mask, other=0.0)
    base = m * Ls * N
    off = rows[:, None] * N + n[None, :]
    m2 = mask[None, :]
    nt = Ls // D
    xt = tl.load(I + base + off, mask=m2, other=0.0)
    for b in range(0, nt):
        xn = tl.load(I + base + (b + 1) * D * N + off, mask=m2 & (b + 1 < nt), other=0.0)
        ut = tl.zeros([D, BN], dtype=tl.float32); wt = tl.zeros([D, BN], dtype=tl.float32); st = tl.zeros([D, BN], dtype=tl.float32)
        for j in tl.static_range(D):
            x = tl.sum(tl.where(rows[:, None] == j, xt, 0.0), axis=0)
            u, w, s = _alif_step(u, w, x, beta, rho, q, theta0, u_r)
            ut = tl.where(rows[:, None] == j, u[None, :], ut)
            wt = tl.where(rows[:, None] == j, w[None, :], wt)
            st = tl.where(rows[:, None] == j, s.to(tl.float32)[None, :], st)
        tl.store(Uout + base + b * D * N + off, ut, mask=m2)
        tl.store(Wout + base + b * D * N + off, wt, mask=m2)
        tl.store(Sout + base + b * D * N + off, st, mask=m2)
        xt = xn
    tl.store(EU + m * N + n, u, mask=mask); tl.store(EW + m * N + n, w, mask=mask)


@triton.jit
def alif_warm_kernel(I, U0, W0, WP, UI, WI, N, C, Wn, u_init, w_init, beta, rho, q, theta0, u_r,
                     BN: tl.constexpr, FROM_STATES: tl.constexpr):
    """Predict the entry adaptation of chunk m by simulating the Wn previous chunks with the current input, from
    (u_init, w_init), or (FROM_STATES) from given chunk-entry states UI/WI (e.g. those of the previous iterate).
    When the warm-up reaches t = 0 it starts from the true initial state, so the first Wn+1 predictions are exact."""
    m = tl.program_id(0); pid = tl.program_id(1)
    n = pid * BN + tl.arange(0, BN); mask = n < N
    s0 = tl.maximum(m - Wn, 0)
    if FROM_STATES:
        ui = tl.load(UI + s0 * N + n, mask=mask, other=0.0); wi = tl.load(WI + s0 * N + n, mask=mask, other=0.0)
    else:
        ui = u_init + tl.zeros([BN], dtype=tl.float32); wi = w_init + tl.zeros([BN], dtype=tl.float32)
    u = tl.where(s0 == 0, tl.load(U0 + n, mask=mask, other=0.0), ui)
    w = tl.where(s0 == 0, tl.load(W0 + n, mask=mask, other=0.0), wi)
    for t in range(s0 * C, m * C):
        x = tl.load(I + t * N + n, mask=mask, other=0.0)
        u, w, s = _alif_step(u, w, x, beta, rho, q, theta0, u_r)
    tl.store(WP + m * N + n, w, mask=mask)


@triton.jit
def alif_build_kernel(I, Lu, Hu, Lw, Hw, MA, MB, LS, LC, LX, NL, POWN, PXL, PXR, PU, PA, POF, NP, OVF,
                      N, C, beta, rho, q, theta0, u_r,
                      K1: tl.constexpr, K2: tl.constexpr, BN: tl.constexpr, LOGK2: tl.constexpr):
    m = tl.program_id(0); pid = tl.program_id(1)
    n = pid * BN + tl.arange(0, BN); mask = n < N
    k1 = tl.arange(0, K1); k2 = tl.arange(0, K2)
    bo = m * N + n
    lu = tl.load(Lu + bo, mask=mask, other=0.0); hu = tl.load(Hu + bo, mask=mask, other=0.0)
    lw = tl.load(Lw + bo, mask=mask, other=0.0); hw = tl.load(Hw + bo, mask=mask, other=0.0)
    A = tl.full([BN], 1.0, tl.float32); B = tl.zeros([BN], dtype=tl.float32); R = tl.full([BN], 1.0, tl.float32)
    wlo = lw
    ls = tl.zeros([BN, K1], dtype=tl.float32); lc = tl.full([BN, K1], float("inf"), tl.float32)
    lx = tl.zeros([BN, K1], dtype=tl.float32)
    nl = tl.zeros([BN], dtype=tl.int32)
    pown = tl.full([BN, K2], -1, tl.int32)
    pxl = tl.full([BN, K2], float("inf"), tl.float32); pxr = tl.full([BN, K2], float("inf"), tl.float32)
    pu = tl.zeros([BN, K2], dtype=tl.float32); pa = tl.full([BN, K2], 1.0, tl.float32); pof = tl.zeros([BN, K2], dtype=tl.float32)
    npc = tl.zeros([BN], dtype=tl.int32)
    ovf = tl.zeros([BN], dtype=tl.int32)
    kk = k2[None, :] + tl.zeros([BN, K2], dtype=tl.int32)
    for j in range(C):
        x = tl.load(I + (m * C + j) * N + n, mask=mask, other=0.0)
        # 1. advance the post-spike pieces born at earlier steps; a piece spikes for x <= z
        valid = k2[None, :] < npc[:, None]
        v = beta * pu + x[:, None]
        r = v - theta0 - pof                             # the piece spikes for x <= z = r / pa  (pa > 0)
        es = valid & (r > pa * pxl)
        en = valid & (r < pa * pxr)
        if tl.max((es & en).to(tl.int32)) > 0:
            # some piece is cut by the threshold: split and compact (O(K log K))
            z = r / pa
            nout = es.to(tl.int32) + en.to(tl.int32)
            p = tl.cumsum(nout, axis=1) - nout
            ntot = tl.sum(nout, axis=1)
            ovf = tl.where(ntot > K2, 1, ovf)
            lo_i = tl.zeros([BN, K2], dtype=tl.int32); hi_i = tl.full([BN, K2], K2 - 1, tl.int32)
            for _ in tl.static_range(LOGK2):
                mid = (lo_i + hi_i + 1) // 2
                le = tl.gather(p, mid, axis=1) <= kk
                lo_i = tl.where(le, mid, lo_i); hi_i = tl.where(le, hi_i, mid - 1)
            idx = lo_i
            d = kk - tl.gather(p, idx, axis=1)
            is_s = (d == 0) & (tl.gather(es.to(tl.int32), idx, axis=1) > 0)
            g_xl = tl.gather(pxl, idx, axis=1); g_xr = tl.gather(pxr, idx, axis=1); g_z = tl.gather(z, idx, axis=1)
            g_v = tl.gather(v, idx, axis=1); g_pa = tl.gather(pa, idx, axis=1); g_po = tl.gather(pof, idx, axis=1)
            g_own = tl.gather(pown, idx, axis=1)
            npc = tl.minimum(ntot, K2)
            keep = k2[None, :] < npc[:, None]
            pxl = tl.where(keep, tl.where(is_s, g_xl, tl.maximum(g_xl, g_z)), float("inf"))
            pxr = tl.where(keep, tl.where(is_s, tl.minimum(g_xr, g_z), g_xr), float("inf"))
            pu = tl.where(keep, tl.where(is_s, u_r, g_v), 0.0)
            pa = tl.where(keep, rho * g_pa, 1.0)
            pof = tl.where(keep, tl.where(is_s, rho * g_po + q, rho * g_po), 0.0)
            pown = tl.where(keep, g_own, -1)
        else:
            # no cut: every piece either spikes as a whole or not at all
            pu = tl.where(valid, tl.where(es, u_r, v), pu)
            pof = tl.where(valid, tl.where(es, rho * pof + q, rho * pof), pof)
            pa = tl.where(valid, rho * pa, pa)
        # 2. first-spike line of step j
        A = beta * A; B = beta * B + x
        sj = R / A; cj = (theta0 - B) / A
        whi = tl.minimum(hw, (hu - cj) / sj)
        validl = k1[None, :] < nl[:, None]
        dom = (validl & (ls * wlo[:, None] + lc <= (sj * wlo + cj)[:, None]) & (ls * whi[:, None] + lc <= (sj * whi + cj)[:, None]))
        dom = tl.max(dom.to(tl.int32), axis=1) > 0
        newc = (wlo < whi) & (~dom) & mask
        cx = R * rho                                     # rho^j: x = rho^j w0 + q after a first spike at j
        put = newc[:, None] & (k1[None, :] == nl[:, None])
        ls = tl.where(put, sj[:, None], ls); lc = tl.where(put, cj[:, None], lc); lx = tl.where(put, cx[:, None], lx)
        ovf = tl.where(newc & (nl >= K1), 1, ovf)
        xlo = cx * wlo + q; xhi = cx * whi + q
        pad = 4e-7 * (tl.abs(xhi) + 1.0)                 # widening an x-range is always safe
        putp = newc[:, None] & (k2[None, :] == npc[:, None])
        pxl = tl.where(putp, (xlo - pad)[:, None], pxl); pxr = tl.where(putp, (xhi + pad)[:, None], pxr)
        pu = tl.where(putp, u_r, pu); pa = tl.where(putp, 1.0, pa); pof = tl.where(putp, 0.0, pof)
        pown = tl.where(putp, nl[:, None], pown)
        ovf = tl.where(newc & (npc >= K2), 1, ovf)
        nl = tl.where(newc, tl.minimum(nl + 1, K1), nl)
        npc = tl.where(newc, tl.minimum(npc + 1, K2), npc)
        # states with w0 <= (lu - cj)/sj have u0 >= lu >= g_j(w0): they have spiked by step j
        wlo = tl.maximum(wlo, (lu - cj) / sj)
        R = rho * R
    tl.store(MA + bo, A, mask=mask); tl.store(MB + bo, B, mask=mask)
    tl.store(NL + bo, nl, mask=mask); tl.store(NP + bo, npc, mask=mask); tl.store(OVF + bo, ovf, mask=mask)
    offl = (m * K1 + k1[None, :]) * N + n[:, None]
    tl.store(LS + offl, ls, mask=mask[:, None]); tl.store(LC + offl, lc, mask=mask[:, None]); tl.store(LX + offl, lx, mask=mask[:, None])
    offp = (m * K2 + k2[None, :]) * N + n[:, None]
    tl.store(POWN + offp, pown, mask=mask[:, None]); tl.store(PXL + offp, pxl, mask=mask[:, None])
    tl.store(PXR + offp, pxr, mask=mask[:, None]); tl.store(PU + offp, pu, mask=mask[:, None])
    tl.store(PA + offp, pa, mask=mask[:, None]); tl.store(POF + offp, pof, mask=mask[:, None])


@triton.jit
def _carry_load(m, ok, n, mask, k1, k2, N, Lu, Hu, Lw, Hw, MA, MB, LS, LC, LX, NL, POWN, PXL, PXR, PU, PA, POF, NP, OVF,
                K1: tl.constexpr, K2: tl.constexpr):
    mk = mask & ok
    bo = m * N + n
    lu = tl.load(Lu + bo, mask=mk, other=0.0); hu = tl.load(Hu + bo, mask=mk, other=0.0)
    lw = tl.load(Lw + bo, mask=mk, other=0.0); hw = tl.load(Hw + bo, mask=mk, other=0.0)
    A = tl.load(MA + bo, mask=mk, other=1.0); B = tl.load(MB + bo, mask=mk, other=0.0)
    nl = tl.load(NL + bo, mask=mk, other=0); npc = tl.load(NP + bo, mask=mk, other=0)
    ovf = tl.load(OVF + bo, mask=mk, other=0)
    offl = (m * K1 + k1[None, :]) * N + n[:, None]
    ls = tl.load(LS + offl, mask=mk[:, None], other=0.0)
    lc = tl.load(LC + offl, mask=mk[:, None], other=float("inf"))
    lx = tl.load(LX + offl, mask=mk[:, None], other=0.0)
    offp = (m * K2 + k2[None, :]) * N + n[:, None]
    pown = tl.load(POWN + offp, mask=mk[:, None], other=-1)
    pxl = tl.load(PXL + offp, mask=mk[:, None], other=float("inf"))
    pxr = tl.load(PXR + offp, mask=mk[:, None], other=float("inf"))
    pu = tl.load(PU + offp, mask=mk[:, None], other=0.0)
    pa = tl.load(PA + offp, mask=mk[:, None], other=0.0)
    pof = tl.load(POF + offp, mask=mk[:, None], other=0.0)
    return lu, hu, lw, hw, A, B, nl, npc, ovf, ls, lc, lx, pown, pxl, pxr, pu, pa, pof


@triton.jit
def alif_carry_kernel(I, U0, W0, Lu, Hu, Lw, Hw, MA, MB, LS, LC, LX, NL, POWN, PXL, PXR, PU, PA, POF, NP, OVF,
                      SU, SW, MISS, N, M, C, beta, rho, q, theta0, u_r, rhoC,
                      K1: tl.constexpr, K2: tl.constexpr, BN: tl.constexpr, DF: tl.constexpr):
    """Sequential over chunks.  The map of chunk m+1 is loaded while chunk m is evaluated (it does not depend on
    the state); a chunk whose entry state misses the box is evaluated serially with DF-step input prefetch."""
    pid = tl.program_id(0); n = pid * BN + tl.arange(0, BN); mask = n < N
    k1 = tl.arange(0, K1); k2 = tl.arange(0, K2); rows = tl.arange(0, DF)
    u = tl.load(U0 + n, mask=mask, other=0.0); w = tl.load(W0 + n, mask=mask, other=0.0)
    lu, hu, lw, hw, A, B, nl, npc, ovf, ls, lc, lx, pown, pxl, pxr, pu, pa, pof = _carry_load(
        0, M > 0, n, mask, k1, k2, N, Lu, Hu, Lw, Hw, MA, MB, LS, LC, LX, NL, POWN, PXL, PXR, PU, PA, POF, NP, OVF, K1, K2)
    for m in range(M):
        lu2, hu2, lw2, hw2, A2, B2, nl2, npc2, ovf2, ls2, lc2, lx2, pown2, pxl2, pxr2, pu2, pa2, pof2 = _carry_load(
            m + 1, m + 1 < M, n, mask, k1, k2, N, Lu, Hu, Lw, Hw, MA, MB, LS, LC, LX, NL, POWN, PXL, PXR, PU, PA, POF, NP, OVF, K1, K2)
        bo = m * N + n
        tl.store(SU + bo, u, mask=mask); tl.store(SW + bo, w, mask=mask)
        fire = (k1[None, :] < nl[:, None]) & (u[:, None] >= ls * w[:, None] + lc)
        first = tl.min(tl.where(fire, k1[None, :], K1), axis=1)
        none_ = first == K1
        xq = tl.sum(tl.where(k1[None, :] == first[:, None], lx, 0.0), axis=1) * w + q
        hit = (k2[None, :] < npc[:, None]) & (pown == first[:, None]) & (pxl <= xq[:, None]) & (xq[:, None] < pxr)
        nh = tl.sum(hit.to(tl.int32), axis=1)
        mu = tl.sum(tl.where(hit, pu, 0.0), axis=1)
        mw = tl.sum(tl.where(hit, pa * xq[:, None] + pof, 0.0), axis=1)
        nu = tl.where(none_, A * u + B, mu)
        nw = tl.where(none_, rhoC * w, mw)
        inbox = (u >= lu) & (u < hu) & (w >= lw) & (w <= hw)
        bad = ((~inbox) | (ovf > 0) | ((~none_) & (nh != 1))) & mask
        tl.store(MISS + bo, bad.to(tl.int32), mask=mask)
        if tl.max(bad.to(tl.int32), axis=0) > 0:        # entry state outside the box, overflow or no piece: serial
            uu = u; ww = w
            for blk in range(C // DF):
                xt = tl.load(I + (m * C + blk * DF + rows[:, None]) * N + n[None, :], mask=mask[None, :], other=0.0)
                for jj in tl.static_range(DF):
                    x = tl.sum(tl.where(rows[:, None] == jj, xt, 0.0), axis=0)
                    uu, ww, s_ = _alif_step(uu, ww, x, beta, rho, q, theta0, u_r)
            nu = tl.where(bad, uu, nu); nw = tl.where(bad, ww, nw)
        u = nu; w = nw
        lu = lu2; hu = hu2; lw = lw2; hw = hw2; A = A2; B = B2; nl = nl2; npc = npc2; ovf = ovf2
        ls = ls2; lc = lc2; lx = lx2; pown = pown2; pxl = pxl2; pxr = pxr2; pu = pu2; pa = pa2; pof = pof2


def alif_fwd_cfg(N):
    """(BN, D, num_warps) of the time-tiled serial ALIF forward, per width, from E26b (RTX 5090)."""
    if N <= 8: return 16, 16, 1
    if N <= 64: return 16, 4, 1
    if N <= 4096: return 32, 32, 1
    return 128, 8, 1


def alif_serial_forward(I, beta, rho, q, theta0, u_r, u0, w0, BN=32, D=0, num_warps=1):
    """Serial reference.  D = 0: plain loop; D > 0: time-tiled with D-step prefetch."""
    T, N = I.shape
    U = torch.empty_like(I); W = torch.empty_like(I); S = torch.empty_like(I)
    if D == 0:
        alif_serial_kernel[(triton.cdiv(N, BN),)](I, u0, w0, U, W, S, T, N, beta, rho, q, theta0, u_r, BN=BN, num_warps=num_warps)
    else:
        EU = torch.empty(1, N, device=I.device); EW = torch.empty(1, N, device=I.device)
        alif_tiled_kernel[(1, triton.cdiv(N, BN))](I, u0, w0, U, W, S, EU, EW, N, T, beta, rho, q, theta0, u_r,
                                                   BN=BN, D=D, num_warps=num_warps)
    return U, W, S


def alif_warm_predict(I, beta, rho, q, theta0, u_r, u0, w0, C, Wn=4, u_init=0.0, w_init=0.0, BN=32, UI=None, WI=None):
    """UI, WI: optional [T//C, N] chunk-entry states (e.g. of the previous iterate) to start the warm-up from."""
    T, N = I.shape; M = T // C
    WP = torch.empty(M, N, device=I.device)
    fs = UI is not None
    if not fs:
        UI = WI = WP
    alif_warm_kernel[(M, triton.cdiv(N, BN))](I, u0, w0, WP, UI, WI, N, C, Wn, u_init, w_init, beta, rho, q, theta0, u_r,
                                              BN=BN, FROM_STATES=fs, num_warps=1)
    return WP


class _Phases:
    def __init__(self, on):
        self.on, self.ev = on, []

    def mark(self, name):
        if self.on:
            e = torch.cuda.Event(enable_timing=True); e.record(); self.ev.append((name, e))

    def result(self):
        torch.cuda.synchronize()
        return {self.ev[i + 1][0]: self.ev[i][1].elapsed_time(self.ev[i + 1][1]) for i in range(len(self.ev) - 1)}


def alif_scan_forward(I, beta, rho, q, theta0, u_r, u0, w0, Wpred, dw, C=128, K1=16, K2=32, BN=16, nw=1, BNc=1,
                      BNr=32, D=4, repair=0, stats=None, phases=None):
    """Wpred: [T//C, N] predicted entry adaptation of every chunk; dw: half-width of the w-box.
    phases: optional dict that receives per-phase GPU times (ms)."""
    ph = _Phases(phases is not None); ph.mark("start")
    T, N = I.shape
    assert T % C == 0 and C % D == 0
    M = T // C
    dev = I.device
    EC = torch.empty(M, N, device=dev); EA = torch.empty(M, N, device=dev); EB = torch.empty(M, N, device=dev)
    L.env_chunk_kernel[(M, triton.cdiv(N, BNr))](I, EC, EA, EB, N, C, beta, u_r, BN=BNr, D=1, num_warps=1)
    Lu = torch.empty(M, N, device=dev)
    L.env_carry_kernel[(triton.cdiv(N, 16),)](u0, EC, EA, EB, Lu, N, M, 1e-5, BN=16, num_warps=1)
    ph.mark("envelope")
    Wp = Wpred.clone(); Wp[0] = w0
    Lw = (Wp - dw).clamp(min=0.0).contiguous(); Hw = (Wp + dw).contiguous()
    Hu = torch.clamp(theta0 + Hw / rho, min=u_r)
    Hu = (Hu + 1e-5 * (Hu.abs() + 1.0)).contiguous()
    MA = torch.empty(M, N, device=dev); MB = torch.empty(M, N, device=dev)
    LS = torch.empty(M, K1, N, device=dev); LC = torch.empty(M, K1, N, device=dev); LX = torch.empty(M, K1, N, device=dev)
    NL = torch.empty(M, N, device=dev, dtype=torch.int32); NP = torch.empty(M, N, device=dev, dtype=torch.int32)
    OVF = torch.empty(M, N, device=dev, dtype=torch.int32)
    POWN = torch.empty(M, K2, N, device=dev, dtype=torch.int32)
    PXL = torch.empty(M, K2, N, device=dev); PXR = torch.empty(M, K2, N, device=dev)
    PU = torch.empty(M, K2, N, device=dev); PA = torch.empty(M, K2, N, device=dev); POF = torch.empty(M, K2, N, device=dev)
    alif_build_kernel[(M, triton.cdiv(N, BN))](I, Lu, Hu, Lw, Hw, MA, MB, LS, LC, LX, NL, POWN, PXL, PXR, PU, PA, POF, NP, OVF,
                                               N, C, beta, rho, q, theta0, u_r, K1=K1, K2=K2, BN=BN,
                                               LOGK2=int(math.log2(K2)), num_warps=nw)
    ph.mark("build")
    SU = torch.empty(M, N, device=dev); SW = torch.empty(M, N, device=dev)
    MISS = torch.empty(M, N, device=dev, dtype=torch.int32)
    rhoC = float(torch.tensor(rho, dtype=torch.float32) ** C)
    alif_carry_kernel[(triton.cdiv(N, BNc),)](I, u0, w0, Lu, Hu, Lw, Hw, MA, MB, LS, LC, LX, NL, POWN, PXL, PXR, PU, PA, POF, NP, OVF,
                                              SU, SW, MISS, N, M, C, beta, rho, q, theta0, u_r, rhoC,
                                              K1=K1, K2=K2, BN=BNc, DF=min(8, C), num_warps=1)
    ph.mark("carry")
    U = torch.empty_like(I); W = torch.empty_like(I); S = torch.empty_like(I)
    EU = torch.empty(M, N, device=dev); EW = torch.empty(M, N, device=dev)
    gr = (M, triton.cdiv(N, BNr))
    alif_tiled_kernel[gr](I, SU, SW, U, W, S, EU, EW, N, C, beta, rho, q, theta0, u_r, BN=BNr, D=D, num_warps=1)
    ph.mark("replay")
    it = 0
    for it in range(1, repair + 1):
        nu_ = torch.cat([u0[None], EU[:-1]], 0); nw_ = torch.cat([w0[None], EW[:-1]], 0)
        if torch.equal(nu_, SU) and torch.equal(nw_, SW):
            it -= 1; break
        SU = nu_.contiguous(); SW = nw_.contiguous()
        alif_tiled_kernel[gr](I, SU, SW, U, W, S, EU, EW, N, C, beta, rho, q, theta0, u_r, BN=BNr, D=D, num_warps=1)
    if phases is not None:
        phases.update(ph.result())
    if stats is not None:
        stats["repair_iters"] = it
        stats["miss_frac"] = MISS.float().mean().item()
        stats["overflow_frac"] = OVF.float().mean().item()
        stats["mean_lines"] = NL.float().mean().item()
        stats["mean_pieces"] = NP.float().mean().item()
        stats["max_pieces"] = int(NP.max().item())
    return U, W, S


# ============================================================================= backward (detached reset)
# a_t = v_t - theta0 - w_{t-1},  sg_t = surrogate'(a_t),  g_t = dL/ds_t (external),  d_t = g_t + q*lw_t
#   lw_{t-1} = rho*lw_t - sg_t*d_t = (rho - q*sg_t)*lw_t - sg_t*g_t        (lw_t = dL/dw_t: a 1-D affine recurrence)
#   lv_t     = (1 - s_t)*beta*lv_{t+1} + sg_t*d_t                            (lv_t = dL/dv_t: 1-D affine given lw)
# Both are solved by two-level affine scans (compose chunks, carry, replay), the second one reading the first.
@triton.jit
def _alif_sg(up, wp, x, beta, theta0, alpha):
    v = beta * up + x
    sig = 1.0 / (1.0 + tl.exp(-alpha * (v - theta0 - wp)))
    return alpha * sig * (1.0 - sig)


@triton.jit
def alif_bwd_kernel(I, U0, W0, Uout, Wout, Sout, GS, LW, LV, PA, PB, LamEnd, N, Ls, beta, rho, q, theta0, alpha,
                    PASS: tl.constexpr, BN: tl.constexpr, D: tl.constexpr):
    """Reverse sweep over segment m (steps m*Ls .. m*Ls+Ls-1).
    PASS 0: compose the lw maps (PA, PB);  1: lw replay from LamEnd (writes LW[t] = lw_t);
    PASS 2: compose the lv maps (reads LW);  3: lv replay (writes LV[t] = lv_t);  4: fused serial sweep (both)."""
    m = tl.program_id(0); pid = tl.program_id(1)
    n = pid * BN + tl.arange(0, BN); mask = n < N
    rows = tl.arange(0, D); m2 = mask[None, :]
    u0 = tl.load(U0 + n, mask=mask, other=0.0); w0 = tl.load(W0 + n, mask=mask, other=0.0)
    lam = tl.zeros([BN], dtype=tl.float32)
    if PASS == 1 or PASS == 3:
        lam = tl.load(LamEnd + m * N + n, mask=mask, other=0.0)
    P = tl.full([BN], 1.0, tl.float32); Q = tl.zeros([BN], dtype=tl.float32)
    lw = tl.zeros([BN], dtype=tl.float32); lv = tl.zeros([BN], dtype=tl.float32)
    nt = Ls // D
    for i in range(0, nt):
        b = nt - 1 - i
        t0 = m * Ls + b * D
        off = (t0 + rows[:, None]) * N + n[None, :]
        xt = tl.load(I + off, mask=m2, other=0.0)
        stl = tl.load(Sout + off, mask=m2, other=0.0)
        gst = tl.load(GS + off, mask=m2, other=0.0)
        pm = m2 & (t0 + rows[:, None] > 0)
        first = t0 + rows[:, None] == 0
        upt = tl.where(first, u0[None, :], tl.load(Uout + off - N, mask=pm, other=0.0))
        wpt = tl.where(first, w0[None, :], tl.load(Wout + off - N, mask=pm, other=0.0))
        lwt = tl.zeros([D, BN], dtype=tl.float32)
        if PASS == 2 or PASS == 3:
            lwt = tl.load(LW + off, mask=m2, other=0.0)
        out = tl.zeros([D, BN], dtype=tl.float32); out2 = tl.zeros([D, BN], dtype=tl.float32)
        for jj in tl.static_range(D):
            j = D - 1 - jj
            sel = rows[:, None] == j
            x = tl.sum(tl.where(sel, xt, 0.0), axis=0)
            s = tl.sum(tl.where(sel, stl, 0.0), axis=0)
            gs = tl.sum(tl.where(sel, gst, 0.0), axis=0)
            up = tl.sum(tl.where(sel, upt, 0.0), axis=0)
            wp = tl.sum(tl.where(sel, wpt, 0.0), axis=0)
            sg = _alif_sg(up, wp, x, beta, theta0, alpha)
            if PASS == 0:
                a = rho - q * sg
                P = a * P; Q = a * Q - sg * gs
            elif PASS == 1:
                out = tl.where(sel, lam[None, :], out)
                lam = (rho - q * sg) * lam - sg * gs
            elif PASS == 2:
                lwj = tl.sum(tl.where(sel, lwt, 0.0), axis=0)
                a = (1.0 - s) * beta
                P = a * P; Q = a * Q + sg * (gs + q * lwj)
            elif PASS == 3:
                lwj = tl.sum(tl.where(sel, lwt, 0.0), axis=0)
                lam = (1.0 - s) * beta * lam + sg * (gs + q * lwj)
                out = tl.where(sel, lam[None, :], out)
            else:
                d = gs + q * lw
                lv = (1.0 - s) * beta * lv + sg * d
                out = tl.where(sel, lw[None, :], out)
                out2 = tl.where(sel, lv[None, :], out2)
                lw = rho * lw - sg * d
        if PASS == 1:
            tl.store(LW + off, out, mask=m2)
        if PASS == 3:
            tl.store(LV + off, out, mask=m2)
        if PASS == 4:
            tl.store(LW + off, out, mask=m2); tl.store(LV + off, out2, mask=m2)
    if PASS == 0 or PASS == 2:
        tl.store(PA + m * N + n, P, mask=mask); tl.store(PB + m * N + n, Q, mask=mask)


@triton.jit
def alif_bwd_serial_pf_kernel(I, U0, W0, Uout, Wout, Sout, GS, LW, LV, N, T, beta, rho, q, theta0, alpha,
                              BN: tl.constexpr, D: tl.constexpr):
    """Fused serial reverse sweep (same arithmetic as PASS 4) that loads tile b-1 while computing tile b."""
    pid = tl.program_id(0)
    n = pid * BN + tl.arange(0, BN); mask = n < N
    rows = tl.arange(0, D); m2 = mask[None, :]
    u0 = tl.load(U0 + n, mask=mask, other=0.0); w0 = tl.load(W0 + n, mask=mask, other=0.0)
    lw = tl.zeros([BN], dtype=tl.float32); lv = tl.zeros([BN], dtype=tl.float32)
    nt = T // D
    t0 = (nt - 1) * D
    off = (t0 + rows[:, None]) * N + n[None, :]
    pm = m2 & (t0 + rows[:, None] > 0)
    xt = tl.load(I + off, mask=m2, other=0.0); stl = tl.load(Sout + off, mask=m2, other=0.0)
    gst = tl.load(GS + off, mask=m2, other=0.0)
    upt = tl.load(Uout + off - N, mask=pm, other=0.0); wpt = tl.load(Wout + off - N, mask=pm, other=0.0)
    for i in range(0, nt):
        b = nt - 1 - i
        t0 = b * D
        offn = (t0 - D + rows[:, None]) * N + n[None, :]
        mn = m2 & (b >= 1)
        pmn = mn & (t0 - D + rows[:, None] > 0)
        xn = tl.load(I + offn, mask=mn, other=0.0); sn = tl.load(Sout + offn, mask=mn, other=0.0)
        gsn = tl.load(GS + offn, mask=mn, other=0.0)
        upn = tl.load(Uout + offn - N, mask=pmn, other=0.0); wpn = tl.load(Wout + offn - N, mask=pmn, other=0.0)
        first = t0 + rows[:, None] == 0
        upc = tl.where(first, u0[None, :], upt); wpc = tl.where(first, w0[None, :], wpt)
        out = tl.zeros([D, BN], dtype=tl.float32); out2 = tl.zeros([D, BN], dtype=tl.float32)
        for jj in tl.static_range(D):
            j = D - 1 - jj
            sel = rows[:, None] == j
            x = tl.sum(tl.where(sel, xt, 0.0), axis=0)
            s = tl.sum(tl.where(sel, stl, 0.0), axis=0)
            gs = tl.sum(tl.where(sel, gst, 0.0), axis=0)
            up = tl.sum(tl.where(sel, upc, 0.0), axis=0)
            wp = tl.sum(tl.where(sel, wpc, 0.0), axis=0)
            sg = _alif_sg(up, wp, x, beta, theta0, alpha)
            d = gs + q * lw
            lv = (1.0 - s) * beta * lv + sg * d
            out = tl.where(sel, lw[None, :], out)
            out2 = tl.where(sel, lv[None, :], out2)
            lw = rho * lw - sg * d
        o = (t0 + rows[:, None]) * N + n[None, :]
        tl.store(LW + o, out, mask=m2); tl.store(LV + o, out2, mask=m2)
        xt = xn; stl = sn; gst = gsn; upt = upn; wpt = wpn


def alif_backward(I, U, W, S, GS, beta, rho, q, theta0, u0, w0, alpha=4.0, mode="scan", C=64, BN=32, D=4, nw=1):
    """Returns (LV, LW): LV[t] = dL/dv_t (= dL/dI_t), LW[t] = dL/dw_t, for a loss with dL/dS = GS."""
    T, N = I.shape
    dev = I.device
    LW = torch.empty_like(I); LV = torch.empty_like(I)
    dummy = torch.empty(1, N, device=dev)
    if mode == "serial_pf":
        alif_bwd_serial_pf_kernel[(triton.cdiv(N, BN),)](I, u0, w0, U, W, S, GS, LW, LV, N, T, beta, rho, q, theta0, alpha,
                                                         BN=BN, D=D, num_warps=nw)
        return LV, LW
    if mode == "serial_tuned":
        BN_, D_, nw_ = L.serial_bwd_cfg(N)
        while T % D_: D_ //= 2
        alif_bwd_serial_pf_kernel[(triton.cdiv(N, BN_),)](I, u0, w0, U, W, S, GS, LW, LV, N, T, beta, rho, q, theta0, alpha,
                                                          BN=BN_, D=D_, num_warps=nw_)
        return LV, LW
    if mode == "serial":
        alif_bwd_kernel[(1, triton.cdiv(N, BN))](I, u0, w0, U, W, S, GS, LW, LV, dummy, dummy, dummy, N, T, beta, rho, q, theta0,
                                                 alpha, PASS=4, BN=BN, D=D, num_warps=nw)
        return LV, LW
    assert T % C == 0 and C % D == 0
    M = T // C
    PA = torch.empty(M, N, device=dev); PB = torch.empty(M, N, device=dev); LamEnd = torch.empty(M, N, device=dev)
    gr = (M, triton.cdiv(N, BN)); gc = (triton.cdiv(N, BN),)
    args = (I, u0, w0, U, W, S, GS, LW, LV, PA, PB, LamEnd, N, C, beta, rho, q, theta0, alpha)
    alif_bwd_kernel[gr](*args, PASS=0, BN=BN, D=D, num_warps=nw)
    L.bwd_carry_kernel[gc](PA, PB, LamEnd, N, M, BN=BN, num_warps=1)
    alif_bwd_kernel[gr](*args, PASS=1, BN=BN, D=D, num_warps=nw)
    alif_bwd_kernel[gr](*args, PASS=2, BN=BN, D=D, num_warps=nw)
    L.bwd_carry_kernel[gc](PA, PB, LamEnd, N, M, BN=BN, num_warps=1)
    alif_bwd_kernel[gr](*args, PASS=3, BN=BN, D=D, num_warps=nw)
    return LV, LW


class ALIFFn(torch.autograd.Function):
    """S = ALIF(I; u0, w0, beta, rho, q, theta0) with u_r fixed.  Forward: serial, or the box scan with prediction
    Wpred (float tensor [T//C, N]); backward: serial or scan.  Gradients for I, u0, w0, beta, rho, q, theta0."""

    @staticmethod
    def forward(ctx, I, u0, w0, beta, rho, q, theta0, u_r, alpha, fwd, bwd, Wpred, dw, C, Cb):
        b, r, qq, th = float(beta), float(rho), float(q), float(theta0)
        if fwd == "serial":
            BN_, D_, _ = alif_fwd_cfg(I.shape[1])
            U, W, S = alif_serial_forward(I, b, r, qq, th, u_r, u0, w0, BN=BN_, D=D_)
        else:
            U, W, S = alif_scan_forward(I, b, r, qq, th, u_r, u0, w0, Wpred, dw, C=C, K1=32, K2=32, BN=4, BNc=1, D=4)
        ctx.save_for_backward(I, u0, w0, U, W, S)
        ctx.c = (b, r, qq, th, u_r, alpha, bwd, Cb)
        ctx.mark_non_differentiable(U, W)
        return S, U, W

    @staticmethod
    def backward(ctx, gS, gU, gW):
        I, u0, w0, U, W, S = ctx.saved_tensors
        b, r, qq, th, u_r, alpha, bwd, Cb = ctx.c
        GS = gS.contiguous()
        LV, LW = alif_backward(I, U, W, S, GS, b, r, qq, th, u0, w0, alpha=alpha, mode="serial_tuned" if bwd == "serial" else bwd, C=Cb)
        Up = torch.cat([u0[None], U[:-1]], 0); Wp = torch.cat([w0[None], W[:-1]], 0)
        V = b * Up + I
        sg = alpha * torch.sigmoid(alpha * (V - th - Wp)); sg = sg * (1 - torch.sigmoid(alpha * (V - th - Wp)))
        d = GS + qq * LW
        lw_m1 = r * LW[0] - sg[0] * d[0]                 # dL/dw0
        dev = I.device
        return (LV, b * LV[0], lw_m1, (LV * Up).sum().reshape(()), (LW * Wp).sum().reshape(()), (LW * S).sum().reshape(()),
                -(sg * d).sum().reshape(()), None, None, None, None, None, None, None, None)


def alif(I, u0, w0, beta, rho, q, theta0=1.0, u_r=0.0, alpha=4.0, fwd="serial", bwd="scan", Wpred=None, dw=None, C=128, Cb=64):
    """Differentiable ALIF layer; returns (S, U, W).  Parameters may be 0-d tensors (requires_grad) or floats."""
    t = lambda v: v if torch.is_tensor(v) else torch.tensor(float(v), device=I.device)
    return ALIFFn.apply(I, u0, w0, t(beta), t(rho), t(q), t(theta0), u_r, alpha, fwd, bwd, Wpred, dw, C, Cb)
