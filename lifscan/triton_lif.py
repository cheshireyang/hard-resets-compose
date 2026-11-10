"""Triton kernels: fused serial baseline and the two-level map-composition scan.

Layout: I is [T, N] float32 contiguous (N = batch * neurons), element (t, n) at t*N + n.
All kernels share ``_step`` so the serial baseline and the scan's replay execute the
identical floating-point instruction sequence (needed for bit-exactness).

Forward scan (chunk length C, M = T // C chunks, map capacity K):
  A  build : grid (M, N/BN)  -- fold C steps into a pruned segment map (K slots)
  B  carry : grid (N/BN)     -- evaluate the M maps sequentially on scalars
  C  replay: grid (M, N/BN)  -- replay each chunk from its start state, write u, s
Backward (linear recurrence lam_t = a_t lam_{t+1} + b_t), same three phases.
"""
import math

import torch
import triton
import triton.language as tl


@triton.jit
def _step(u, x, beta, theta, u_r):
    v = beta * u + x
    s = v >= theta
    u_new = tl.where(s, u_r, v)
    return u_new, s, v


# ----------------------------------------------------------------------------- serial
@triton.jit
def serial_fwd_kernel(I, U0, Uout, Sout, T, N, beta, theta, u_r, BN: tl.constexpr):
    pid = tl.program_id(0)
    n = pid * BN + tl.arange(0, BN)
    mask = n < N
    u = tl.load(U0 + n, mask=mask, other=0.0)
    for t in range(T):
        x = tl.load(I + t * N + n, mask=mask, other=0.0)
        u, s, v = _step(u, x, beta, theta, u_r)
        tl.store(Uout + t * N + n, u, mask=mask)
        tl.store(Sout + t * N + n, s.to(tl.float32), mask=mask)


@triton.jit
def _coef(u_prev, s, x, gs, gu, beta, theta, u_r, alpha, DETACH: tl.constexpr):
    v = beta * u_prev + x
    sig = 1.0 / (1.0 + tl.exp(-alpha * (v - theta)))
    sg = alpha * sig * (1.0 - sig)
    J = 1.0 - s
    if not DETACH:
        J = J + (u_r - v) * sg
    return beta * J, gs * sg + J * gu


@triton.jit
def serial_bwd_kernel(I, U0, Uout, Sout, GS, GU, Lam, T, N, beta, theta, u_r, alpha,
                      DETACH: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)
    n = pid * BN + tl.arange(0, BN)
    mask = n < N
    lam = tl.zeros([BN], dtype=tl.float32)
    for i in range(T):
        t = T - 1 - i
        x = tl.load(I + t * N + n, mask=mask, other=0.0)
        s = tl.load(Sout + t * N + n, mask=mask, other=0.0)
        if t > 0:
            up = tl.load(Uout + (t - 1) * N + n, mask=mask, other=0.0)
        else:
            up = tl.load(U0 + n, mask=mask, other=0.0)
        gs = tl.load(GS + t * N + n, mask=mask, other=0.0)
        gu = tl.load(GU + t * N + n, mask=mask, other=0.0)
        a, b = _coef(up, s, x, gs, gu, beta, theta, u_r, alpha, DETACH)
        lam = b + a * lam  # a here multiplies lam_{t+1}; at t = T-1, lam = 0
        tl.store(Lam + t * N + n, lam, mask=mask)


# ----------------------------------------------------------------------------- scan fwd
@triton.jit
def build_kernel(I, Lo, Hi, MA, MB, MX, MC, MN, OVF, N, C, beta, theta, u_r,
                 K: tl.constexpr, BN: tl.constexpr):
    m = tl.program_id(0)
    pid = tl.program_id(1)
    n = pid * BN + tl.arange(0, BN)
    mask = n < N
    ks = tl.arange(0, K)
    lo = tl.load(Lo + n, mask=mask, other=0.0)
    hi = tl.load(Hi + n, mask=mask, other=0.0)
    A = tl.full([BN], 1.0, tl.float32)
    B = tl.zeros([BN], dtype=tl.float32)
    xd = tl.full([BN, K], float("inf"), tl.float32)
    cd = tl.zeros([BN, K], dtype=tl.float32)
    cnt = tl.zeros([BN], dtype=tl.int32)
    xmin = tl.full([BN], float("inf"), tl.float32)
    ovf = tl.zeros([BN], dtype=tl.int32)
    for j in range(C):
        t = m * C + j
        x = tl.load(I + t * N + n, mask=mask, other=0.0)
        vv = beta * cd + x[:, None]
        cd = tl.where(vv >= theta, u_r, vv)
        num = theta - x - beta * B
        den = beta * A
        z = tl.where(den == 0.0, tl.where(num > 0.0, float("inf"), -float("inf")), num / den)
        new = (z < xmin) & (z <= hi) & (xmin > lo)
        ovf = tl.where(new & (cnt >= K), 1, ovf)
        new = new & (cnt < K)
        slot = (ks[None, :] == cnt[:, None]) & new[:, None]
        xd = tl.where(slot, z[:, None], xd)
        cd = tl.where(slot, u_r, cd)
        cnt = cnt + new.to(tl.int32)
        xmin = tl.where(new, z, xmin)
        A = beta * A
        B = beta * B + x
    base = m * N + n
    tl.store(MA + base, A, mask=mask)
    tl.store(MB + base, B, mask=mask)
    tl.store(MN + base, cnt, mask=mask)
    tl.store(OVF + base, ovf, mask=mask)
    off = (m * K + ks[None, :]) * N + n[:, None]
    tl.store(MX + off, xd, mask=mask[:, None])
    tl.store(MC + off, cd, mask=mask[:, None])


@triton.jit
def carry_kernel(I, U0, MA, MB, MX, MC, MN, OVF, Starts, N, M, C, beta, theta, u_r,
                 K: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)
    n = pid * BN + tl.arange(0, BN)
    mask = n < N
    ks = tl.arange(0, K)
    u = tl.load(U0 + n, mask=mask, other=0.0)
    for m in range(M):
        tl.store(Starts + m * N + n, u, mask=mask)
        base = m * N + n
        A = tl.load(MA + base, mask=mask, other=1.0)
        B = tl.load(MB + base, mask=mask, other=0.0)
        cnt = tl.load(MN + base, mask=mask, other=0)
        ovf = tl.load(OVF + base, mask=mask, other=0)
        off = (m * K + ks[None, :]) * N + n[:, None]
        xd = tl.load(MX + off, mask=mask[:, None], other=float("inf"))
        cd = tl.load(MC + off, mask=mask[:, None], other=0.0)
        valid = ks[None, :] < cnt[:, None]
        k = tl.sum(((xd > u[:, None]) & valid).to(tl.int32), axis=1)
        cval = tl.sum(tl.where(ks[None, :] == k[:, None], cd, 0.0), axis=1)
        nu = tl.where(k == cnt, A * u + B, cval)
        if tl.max(ovf, axis=0) > 0:  # rare: serial fallback for this chunk
            w = u
            for j in range(C):
                x = tl.load(I + (m * C + j) * N + n, mask=mask, other=0.0)
                w, s_, v_ = _step(w, x, beta, theta, u_r)
            nu = tl.where(ovf > 0, w, nu)
        u = nu


@triton.jit
def replay_kernel(I, Starts, Uout, Sout, Ends, N, C, beta, theta, u_r, BN: tl.constexpr):
    m = tl.program_id(0)
    pid = tl.program_id(1)
    n = pid * BN + tl.arange(0, BN)
    mask = n < N
    u = tl.load(Starts + m * N + n, mask=mask, other=0.0)
    for j in range(C):
        t = m * C + j
        x = tl.load(I + t * N + n, mask=mask, other=0.0)
        u, s, v = _step(u, x, beta, theta, u_r)
        tl.store(Uout + t * N + n, u, mask=mask)
        tl.store(Sout + t * N + n, s.to(tl.float32), mask=mask)
    tl.store(Ends + m * N + n, u, mask=mask)


# ----------------------------------------------------------------------------- scan bwd
@triton.jit
def bwd_chunk_kernel(I, U0, Uout, Sout, GS, GU, PA, PB, N, C, beta, theta, u_r, alpha,
                     DETACH: tl.constexpr, BN: tl.constexpr):
    """Compose the chunk's affine maps lam_start = P * lam_end + Q (reverse order)."""
    m = tl.program_id(0)
    pid = tl.program_id(1)
    n = pid * BN + tl.arange(0, BN)
    mask = n < N
    P = tl.full([BN], 1.0, tl.float32)
    Q = tl.zeros([BN], dtype=tl.float32)
    for i in range(C):
        t = m * C + C - 1 - i
        x = tl.load(I + t * N + n, mask=mask, other=0.0)
        s = tl.load(Sout + t * N + n, mask=mask, other=0.0)
        if t > 0:
            up = tl.load(Uout + (t - 1) * N + n, mask=mask, other=0.0)
        else:
            up = tl.load(U0 + n, mask=mask, other=0.0)
        gs = tl.load(GS + t * N + n, mask=mask, other=0.0)
        gu = tl.load(GU + t * N + n, mask=mask, other=0.0)
        a, b = _coef(up, s, x, gs, gu, beta, theta, u_r, alpha, DETACH)
        # lam_t = a * lam_{t+1} + b, with lam_{t+1} = P * lam_end + Q
        Q = b + a * Q
        P = a * P
    tl.store(PA + m * N + n, P, mask=mask)
    tl.store(PB + m * N + n, Q, mask=mask)


@triton.jit
def bwd_carry_kernel(PA, PB, LamEnd, N, M, BN: tl.constexpr):
    pid = tl.program_id(0)
    n = pid * BN + tl.arange(0, BN)
    mask = n < N
    lam = tl.zeros([BN], dtype=tl.float32)
    for i in range(M):
        m = M - 1 - i
        tl.store(LamEnd + m * N + n, lam, mask=mask)  # lam_{(m+1)C}
        P = tl.load(PA + m * N + n, mask=mask, other=1.0)
        Q = tl.load(PB + m * N + n, mask=mask, other=0.0)
        lam = Q + P * lam


@triton.jit
def bwd_replay_kernel(I, U0, Uout, Sout, GS, GU, LamEnd, Lam, N, C, beta, theta, u_r, alpha,
                      DETACH: tl.constexpr, BN: tl.constexpr):
    m = tl.program_id(0)
    pid = tl.program_id(1)
    n = pid * BN + tl.arange(0, BN)
    mask = n < N
    lam = tl.load(LamEnd + m * N + n, mask=mask, other=0.0)
    for i in range(C):
        t = m * C + C - 1 - i
        x = tl.load(I + t * N + n, mask=mask, other=0.0)
        s = tl.load(Sout + t * N + n, mask=mask, other=0.0)
        if t > 0:
            up = tl.load(Uout + (t - 1) * N + n, mask=mask, other=0.0)
        else:
            up = tl.load(U0 + n, mask=mask, other=0.0)
        gs = tl.load(GS + t * N + n, mask=mask, other=0.0)
        gu = tl.load(GU + t * N + n, mask=mask, other=0.0)
        a, b = _coef(up, s, x, gs, gu, beta, theta, u_r, alpha, DETACH)
        lam = b + a * lam
        tl.store(Lam + t * N + n, lam, mask=mask)


# ----------------------------------------------------------------------------- python API
BN_DEFAULT = 128


def state_bounds(I, beta, theta, u_r, u0):
    Imin = I.min(0).values.clamp(max=0)
    base = torch.minimum(u0, torch.full_like(u0, u_r)).clamp(max=0)
    lo = base + (Imin / (1 - beta) if beta < 1 else I.clamp(max=0).sum(0))
    hi = torch.maximum(u0, torch.full_like(u0, max(theta, u_r)))
    return lo.contiguous(), hi.contiguous()


def serial_forward(I, beta, theta, u_r, u0, BN=BN_DEFAULT):
    T, N = I.shape
    U = torch.empty_like(I)
    S = torch.empty_like(I)
    serial_fwd_kernel[(triton.cdiv(N, BN),)](I, u0, U, S, T, N, beta, theta, u_r, BN=BN)
    return U, S


def serial_backward(I, u0, U, S, GS, GU, beta, theta, u_r, alpha=4.0, detach=True, BN=BN_DEFAULT):
    T, N = I.shape
    Lam = torch.empty_like(I)
    serial_bwd_kernel[(triton.cdiv(N, BN),)](I, u0, U, S, GS, GU, Lam, T, N, beta, theta, u_r,
                                            alpha, DETACH=detach, BN=BN)
    return Lam


def scan_forward(I, beta, theta, u_r, u0, C=64, K=16, BN=BN_DEFAULT, repair=0, stats=None):
    T, N = I.shape
    assert T % C == 0
    M = T // C
    dev = I.device
    lo, hi = state_bounds(I, beta, theta, u_r, u0)
    MA = torch.empty(M, N, device=dev)
    MB = torch.empty(M, N, device=dev)
    MX = torch.empty(M, K, N, device=dev)
    MC = torch.empty(M, K, N, device=dev)
    MN = torch.empty(M, N, device=dev, dtype=torch.int32)
    OVF = torch.empty(M, N, device=dev, dtype=torch.int32)
    g2 = (M, triton.cdiv(N, BN))
    build_kernel[g2](I, lo, hi, MA, MB, MX, MC, MN, OVF, N, C, beta, theta, u_r, K=K, BN=BN)
    Starts = torch.empty(M, N, device=dev)
    carry_kernel[(triton.cdiv(N, BN),)](I, u0, MA, MB, MX, MC, MN, OVF, Starts, N, M, C,
                                        beta, theta, u_r, K=K, BN=BN)
    U = torch.empty_like(I)
    S = torch.empty_like(I)
    Ends = torch.empty(M, N, device=dev)
    replay_kernel[g2](I, Starts, U, S, Ends, N, C, beta, theta, u_r, BN=BN)
    it = 0
    for it in range(1, repair + 1):
        new = torch.cat([u0[None], Ends[:-1]], 0)
        if torch.equal(new, Starts):
            it -= 1
            break
        Starts = new.contiguous()
        replay_kernel[g2](I, Starts, U, S, Ends, N, C, beta, theta, u_r, BN=BN)
    if stats is not None:
        stats["repair_iters"] = it
        stats["overflow_frac"] = OVF.float().mean().item()
        stats["max_pieces"] = int(MN.max().item()) + 1
    return U, S


def scan_backward(I, u0, U, S, GS, GU, beta, theta, u_r, alpha=4.0, detach=True, C=64, BN=BN_DEFAULT):
    T, N = I.shape
    M = T // C
    dev = I.device
    PA = torch.empty(M, N, device=dev)
    PB = torch.empty(M, N, device=dev)
    g2 = (M, triton.cdiv(N, BN))
    bwd_chunk_kernel[g2](I, u0, U, S, GS, GU, PA, PB, N, C, beta, theta, u_r, alpha,
                         DETACH=detach, BN=BN)
    LamEnd = torch.empty(M, N, device=dev)
    bwd_carry_kernel[(triton.cdiv(N, BN),)](PA, PB, LamEnd, N, M, BN=BN)
    Lam = torch.empty_like(I)
    bwd_replay_kernel[g2](I, u0, U, S, GS, GU, LamEnd, Lam, N, C, beta, theta, u_r, alpha,
                          DETACH=detach, BN=BN)
    return Lam


class LIFFunction(torch.autograd.Function):
    """Hard-reset LIF layer.  mode in {"serial", "scan"}; returns spikes S [T, N]."""

    @staticmethod
    def forward(ctx, I, u0, beta, theta, u_r, alpha, mode, C, K, repair):
        I = I.contiguous()
        if mode == "serial":
            U, S = serial_forward(I, beta, theta, u_r, u0)
        else:
            U, S = scan_forward(I, beta, theta, u_r, u0, C=C, K=K, repair=repair)
        ctx.save_for_backward(I, u0, U, S)
        ctx.cfg = (beta, theta, u_r, alpha, mode, C)
        return S

    @staticmethod
    def backward(ctx, gS):
        I, u0, U, S = ctx.saved_tensors
        beta, theta, u_r, alpha, mode, C = ctx.cfg
        gS = gS.contiguous()
        GU = torch.zeros_like(gS)
        if mode == "serial":
            Lam = serial_backward(I, u0, U, S, gS, GU, beta, theta, u_r, alpha)
        else:
            Lam = scan_backward(I, u0, U, S, gS, GU, beta, theta, u_r, alpha, C=C)
        return Lam, beta * Lam[0], None, None, None, None, None, None, None, None


def lif(I, u0, beta=0.9, theta=1.0, u_r=0.0, alpha=4.0, mode="scan", C=64, K=16, repair=0):
    return LIFFunction.apply(I, u0, beta, theta, u_r, alpha, mode, C, K, repair)


# ============================================================================= tiled (prefetching) kernels
# A chunk of ``Ls`` steps is processed in tiles of D time steps: the next tile's
# inputs are loaded while the current tile is being computed, which hides global
# memory latency in the latency-bound (small N) regime.  The serial baseline is the
# special case of a single chunk (M = 1, Ls = T); the scan's replay uses the very
# same kernel with M = T / C chunks, so both run identical arithmetic.

@triton.jit
def fwd_tiled_kernel(I, Starts, Uout, Sout, Ends, N, Ls, beta, theta, u_r,
                     BN: tl.constexpr, D: tl.constexpr):
    m = tl.program_id(0)
    pid = tl.program_id(1)
    n = pid * BN + tl.arange(0, BN)
    mask = n < N
    rows = tl.arange(0, D)
    u = tl.load(Starts + m * N + n, mask=mask, other=0.0)
    base = m * Ls * N
    off = rows[:, None] * N + n[None, :]
    m2 = mask[None, :]
    nt = Ls // D
    xt = tl.load(I + base + off, mask=m2, other=0.0)
    for b in range(0, nt):
        xn = tl.load(I + base + (b + 1) * D * N + off, mask=m2 & (b + 1 < nt), other=0.0)
        ut = tl.zeros([D, BN], dtype=tl.float32)
        st = tl.zeros([D, BN], dtype=tl.float32)
        for j in tl.static_range(D):
            x = tl.sum(tl.where(rows[:, None] == j, xt, 0.0), axis=0)
            u, s, v = _step(u, x, beta, theta, u_r)
            ut = tl.where(rows[:, None] == j, u[None, :], ut)
            st = tl.where(rows[:, None] == j, s.to(tl.float32)[None, :], st)
        tl.store(Uout + base + b * D * N + off, ut, mask=m2)
        tl.store(Sout + base + b * D * N + off, st, mask=m2)
        xt = xn
    tl.store(Ends + m * N + n, u, mask=mask)


@triton.jit
def bwd_tiled_kernel(I, U0, Uout, Sout, GS, GU, LamEnd, Lam, N, Ls, beta, theta, u_r, alpha,
                     DETACH: tl.constexpr, BN: tl.constexpr, D: tl.constexpr):
    """Reverse sweep over chunk m (steps m*Ls .. m*Ls+Ls-1) starting from LamEnd[m]."""
    m = tl.program_id(0)
    pid = tl.program_id(1)
    n = pid * BN + tl.arange(0, BN)
    mask = n < N
    rows = tl.arange(0, D)
    m2 = mask[None, :]
    lam = tl.load(LamEnd + m * N + n, mask=mask, other=0.0)
    u0 = tl.load(U0 + n, mask=mask, other=0.0)
    nt = Ls // D
    for i in range(0, nt):
        b = nt - 1 - i
        t0 = m * Ls + b * D
        off = (t0 + rows[:, None]) * N + n[None, :]
        xt = tl.load(I + off, mask=m2, other=0.0)
        stl = tl.load(Sout + off, mask=m2, other=0.0)
        gst = tl.load(GS + off, mask=m2, other=0.0)
        gut = tl.load(GU + off, mask=m2, other=0.0)
        upm = m2 & (t0 + rows[:, None] > 0)
        upt = tl.load(Uout + off - N, mask=upm, other=0.0)
        upt = tl.where(t0 + rows[:, None] == 0, u0[None, :], upt)
        lt = tl.zeros([D, BN], dtype=tl.float32)
        for jj in tl.static_range(D):
            j = D - 1 - jj
            sel = rows[:, None] == j
            x = tl.sum(tl.where(sel, xt, 0.0), axis=0)
            s = tl.sum(tl.where(sel, stl, 0.0), axis=0)
            gs = tl.sum(tl.where(sel, gst, 0.0), axis=0)
            gu = tl.sum(tl.where(sel, gut, 0.0), axis=0)
            up = tl.sum(tl.where(sel, upt, 0.0), axis=0)
            a, bb = _coef(up, s, x, gs, gu, beta, theta, u_r, alpha, DETACH)
            lam = bb + a * lam
            lt = tl.where(sel, lam[None, :], lt)
        tl.store(Lam + off, lt, mask=m2)


@triton.jit
def bwd_chunk_tiled_kernel(I, U0, Uout, Sout, GS, GU, PA, PB, N, C, beta, theta, u_r, alpha,
                           DETACH: tl.constexpr, BN: tl.constexpr, D: tl.constexpr):
    m = tl.program_id(0)
    pid = tl.program_id(1)
    n = pid * BN + tl.arange(0, BN)
    mask = n < N
    rows = tl.arange(0, D)
    m2 = mask[None, :]
    u0 = tl.load(U0 + n, mask=mask, other=0.0)
    P = tl.full([BN], 1.0, tl.float32)
    Q = tl.zeros([BN], dtype=tl.float32)
    nt = C // D
    for i in range(0, nt):
        b = nt - 1 - i
        t0 = m * C + b * D
        off = (t0 + rows[:, None]) * N + n[None, :]
        xt = tl.load(I + off, mask=m2, other=0.0)
        stl = tl.load(Sout + off, mask=m2, other=0.0)
        gst = tl.load(GS + off, mask=m2, other=0.0)
        gut = tl.load(GU + off, mask=m2, other=0.0)
        upm = m2 & (t0 + rows[:, None] > 0)
        upt = tl.load(Uout + off - N, mask=upm, other=0.0)
        upt = tl.where(t0 + rows[:, None] == 0, u0[None, :], upt)
        for jj in tl.static_range(D):
            j = D - 1 - jj
            sel = rows[:, None] == j
            x = tl.sum(tl.where(sel, xt, 0.0), axis=0)
            s = tl.sum(tl.where(sel, stl, 0.0), axis=0)
            gs = tl.sum(tl.where(sel, gst, 0.0), axis=0)
            gu = tl.sum(tl.where(sel, gut, 0.0), axis=0)
            up = tl.sum(tl.where(sel, upt, 0.0), axis=0)
            a, bb = _coef(up, s, x, gs, gu, beta, theta, u_r, alpha, DETACH)
            Q = bb + a * Q
            P = a * P
    tl.store(PA + m * N + n, P, mask=mask)
    tl.store(PB + m * N + n, Q, mask=mask)


def serial_forward_tiled(I, beta, theta, u_r, u0, BN=32, D=4, num_warps=1):
    T, N = I.shape
    U = torch.empty_like(I); S = torch.empty_like(I)
    E = torch.empty(1, N, device=I.device)
    fwd_tiled_kernel[(1, triton.cdiv(N, BN))](I, u0, U, S, E, N, T, beta, theta, u_r,
                                              BN=BN, D=D, num_warps=num_warps)
    return U, S


def serial_backward_tiled(I, u0, U, S, GS, GU, beta, theta, u_r, alpha=4.0, detach=True,
                          BN=32, D=4, num_warps=1):
    T, N = I.shape
    Lam = torch.empty_like(I)
    z = torch.zeros(1, N, device=I.device)
    bwd_tiled_kernel[(1, triton.cdiv(N, BN))](I, u0, U, S, GS, GU, z, Lam, N, T, beta, theta, u_r,
                                              alpha, DETACH=detach, BN=BN, D=D, num_warps=num_warps)
    return Lam


def scan_forward_tiled(I, beta, theta, u_r, u0, C=64, K=16, BN=128, BNr=32, D=4, nw=4, nwr=1,
                       repair=0, stats=None):
    """Build/carry as before (BN, nw); replay with the tiled kernel (BNr, D, nwr)."""
    T, N = I.shape
    M = T // C
    dev = I.device
    lo, hi = state_bounds(I, beta, theta, u_r, u0)
    MA = torch.empty(M, N, device=dev); MB = torch.empty(M, N, device=dev)
    MX = torch.empty(M, K, N, device=dev); MC = torch.empty(M, K, N, device=dev)
    MN = torch.empty(M, N, device=dev, dtype=torch.int32)
    OVF = torch.empty(M, N, device=dev, dtype=torch.int32)
    build_kernel[(M, triton.cdiv(N, BN))](I, lo, hi, MA, MB, MX, MC, MN, OVF, N, C, beta, theta, u_r,
                                          K=K, BN=BN, num_warps=nw)
    Starts = torch.empty(M, N, device=dev)
    carry_kernel[(triton.cdiv(N, BN),)](I, u0, MA, MB, MX, MC, MN, OVF, Starts, N, M, C,
                                        beta, theta, u_r, K=K, BN=BN, num_warps=nw)
    U = torch.empty_like(I); S = torch.empty_like(I); Ends = torch.empty(M, N, device=dev)
    gr = (M, triton.cdiv(N, BNr))
    fwd_tiled_kernel[gr](I, Starts, U, S, Ends, N, C, beta, theta, u_r, BN=BNr, D=D, num_warps=nwr)
    it = 0
    for it in range(1, repair + 1):
        new = torch.cat([u0[None], Ends[:-1]], 0)
        if torch.equal(new, Starts):
            it -= 1
            break
        Starts = new.contiguous()
        fwd_tiled_kernel[gr](I, Starts, U, S, Ends, N, C, beta, theta, u_r, BN=BNr, D=D, num_warps=nwr)
    if stats is not None:
        stats["repair_iters"] = it
        stats["overflow_frac"] = OVF.float().mean().item()
    return U, S


def scan_backward_tiled(I, u0, U, S, GS, GU, beta, theta, u_r, alpha=4.0, detach=True, C=64,
                        BNr=32, D=4, nwr=1):
    T, N = I.shape
    M = T // C
    dev = I.device
    PA = torch.empty(M, N, device=dev); PB = torch.empty(M, N, device=dev)
    gr = (M, triton.cdiv(N, BNr))
    bwd_chunk_tiled_kernel[gr](I, u0, U, S, GS, GU, PA, PB, N, C, beta, theta, u_r, alpha,
                               DETACH=detach, BN=BNr, D=D, num_warps=nwr)
    LamEnd = torch.empty(M, N, device=dev)
    bwd_carry_kernel[(triton.cdiv(N, BNr),)](PA, PB, LamEnd, N, M, BN=BNr, num_warps=nwr)
    Lam = torch.empty_like(I)
    bwd_tiled_kernel[gr](I, u0, U, S, GS, GU, LamEnd, Lam, N, C, beta, theta, u_r, alpha,
                         DETACH=detach, BN=BNr, D=D, num_warps=nwr)
    return Lam


# ============================================================================= v2: software-pipelined build/carry
@triton.jit
def build2_kernel(I, Imin, U0, MA, MB, MX, MC, MN, OVF, N, C, beta, theta, u_r, inv1mb, sum_mode,
                  K: tl.constexpr, BN: tl.constexpr, STAGES: tl.constexpr):
    m = tl.program_id(0)
    pid = tl.program_id(1)
    n = pid * BN + tl.arange(0, BN)
    mask = n < N
    ks = tl.arange(0, K)
    u0 = tl.load(U0 + n, mask=mask, other=0.0)
    imin = tl.minimum(tl.load(Imin + n, mask=mask, other=0.0), 0.0)
    base = tl.minimum(tl.minimum(u0, u_r), 0.0)
    lo = base + imin * inv1mb          # inv1mb = 1/(1-beta)  (or T when beta == 1: loose bound)
    hi = tl.maximum(u0, tl.maximum(theta, u_r))
    A = tl.full([BN], 1.0, tl.float32)
    B = tl.zeros([BN], dtype=tl.float32)
    xd = tl.full([BN, K], float("inf"), tl.float32)
    cd = tl.zeros([BN, K], dtype=tl.float32)
    cnt = tl.zeros([BN], dtype=tl.int32)
    xmin = tl.full([BN], float("inf"), tl.float32)
    ovf = tl.zeros([BN], dtype=tl.int32)
    for j in tl.range(0, C, num_stages=STAGES):
        t = m * C + j
        x = tl.load(I + t * N + n, mask=mask, other=0.0)
        vv = beta * cd + x[:, None]
        cd = tl.where(vv >= theta, u_r, vv)
        num = theta - x - beta * B
        den = beta * A
        z = tl.where(den == 0.0, tl.where(num > 0.0, float("inf"), -float("inf")), num / den)
        new = (z < xmin) & (z <= hi) & (xmin > lo)
        ovf = tl.where(new & (cnt >= K), 1, ovf)
        new = new & (cnt < K)
        slot = (ks[None, :] == cnt[:, None]) & new[:, None]
        xd = tl.where(slot, z[:, None], xd)
        cd = tl.where(slot, u_r, cd)
        cnt = cnt + new.to(tl.int32)
        xmin = tl.where(new, z, xmin)
        A = beta * A
        B = beta * B + x
    bo = m * N + n
    tl.store(MA + bo, A, mask=mask)
    tl.store(MB + bo, B, mask=mask)
    tl.store(MN + bo, cnt, mask=mask)
    tl.store(OVF + bo, ovf, mask=mask)
    off = (m * K + ks[None, :]) * N + n[:, None]
    tl.store(MX + off, xd, mask=mask[:, None])
    tl.store(MC + off, cd, mask=mask[:, None])


@triton.jit
def carry2_kernel(I, U0, MA, MB, MX, MC, MN, OVF, Starts, N, M, C, beta, theta, u_r,
                  K: tl.constexpr, BN: tl.constexpr, STAGES: tl.constexpr):
    pid = tl.program_id(0)
    n = pid * BN + tl.arange(0, BN)
    mask = n < N
    ks = tl.arange(0, K)
    u = tl.load(U0 + n, mask=mask, other=0.0)
    for m in tl.range(0, M, num_stages=STAGES):
        tl.store(Starts + m * N + n, u, mask=mask)
        bo = m * N + n
        A = tl.load(MA + bo, mask=mask, other=1.0)
        B = tl.load(MB + bo, mask=mask, other=0.0)
        cnt = tl.load(MN + bo, mask=mask, other=0)
        ovf = tl.load(OVF + bo, mask=mask, other=0)
        off = (m * K + ks[None, :]) * N + n[:, None]
        xd = tl.load(MX + off, mask=mask[:, None], other=float("inf"))
        cd = tl.load(MC + off, mask=mask[:, None], other=0.0)
        valid = ks[None, :] < cnt[:, None]
        k = tl.sum(((xd > u[:, None]) & valid).to(tl.int32), axis=1)
        cval = tl.sum(tl.where(ks[None, :] == k[:, None], cd, 0.0), axis=1)
        nu = tl.where(k == cnt, A * u + B, cval)
        if tl.max(ovf, axis=0) > 0:
            w = u
            for j in range(C):
                x = tl.load(I + (m * C + j) * N + n, mask=mask, other=0.0)
                w, s_, v_ = _step(w, x, beta, theta, u_r)
            nu = tl.where(ovf > 0, w, nu)
        u = nu


def scan_forward_v2(I, beta, theta, u_r, u0, C=64, K=16, BN=128, nw=4, STAGES=3, CSTAGES=1,
                    BNr=32, D=4, nwr=1, repair=0, stats=None, Imin=None):
    """Same algorithm as scan_forward_tiled; fewer launches, pipelined loops.
    Imin (per-neuron min input) may be supplied by the producer of I."""
    T, N = I.shape
    M = T // C
    dev = I.device
    if Imin is None:
        Imin = I.amin(0)
    inv = 1.0 / (1.0 - beta) if beta < 1 else float(T)
    MA = torch.empty(M, N, device=dev); MB = torch.empty(M, N, device=dev)
    MX = torch.empty(M, K, N, device=dev); MC = torch.empty(M, K, N, device=dev)
    MN = torch.empty(M, N, device=dev, dtype=torch.int32)
    OVF = torch.empty(M, N, device=dev, dtype=torch.int32)
    build2_kernel[(M, triton.cdiv(N, BN))](I, Imin, u0, MA, MB, MX, MC, MN, OVF, N, C, beta, theta, u_r,
                                           inv, 0, K=K, BN=BN, STAGES=STAGES, num_warps=nw)
    Starts = torch.empty(M, N, device=dev)
    carry2_kernel[(triton.cdiv(N, BN),)](I, u0, MA, MB, MX, MC, MN, OVF, Starts, N, M, C,
                                         beta, theta, u_r, K=K, BN=BN, STAGES=CSTAGES, num_warps=nw)
    U = torch.empty_like(I); S = torch.empty_like(I); Ends = torch.empty(M, N, device=dev)
    gr = (M, triton.cdiv(N, BNr))
    fwd_tiled_kernel[gr](I, Starts, U, S, Ends, N, C, beta, theta, u_r, BN=BNr, D=D, num_warps=nwr)
    it = 0
    for it in range(1, repair + 1):
        new = torch.cat([u0[None], Ends[:-1]], 0)
        if torch.equal(new, Starts):
            it -= 1
            break
        Starts = new.contiguous()
        fwd_tiled_kernel[gr](I, Starts, U, S, Ends, N, C, beta, theta, u_r, BN=BNr, D=D, num_warps=nwr)
    if stats is not None:
        stats["repair_iters"] = it
        stats["overflow_frac"] = OVF.float().mean().item()
    return U, S



# ============================================================================= tuned autograd layer
def _pick_C(T, D=8):
    for C in (256, 128, 64, 32, 16, 8):
        if T % C == 0 and C % D == 0 and C * C <= 4 * T:
            return C
    raise ValueError(f"T={T} needs a chunk length divisible by {D}")


class LIFTuned(torch.autograd.Function):
    """Hard-reset LIF (detach_reset=True, sigmoid surrogate).  mode in {"serial", "scan"}.
    serial = tuned fused serial Triton kernels; scan = map-composition forward + affine-scan
    backward.  Both produce bit-identical spikes (repair on) and BPTT-equivalent gradients."""

    @staticmethod
    def forward(ctx, I, u0, beta, theta, u_r, alpha, mode, K, repair):
        I = I.contiguous()
        T = I.shape[0]
        if mode == "serial":
            U, S = serial_forward_tiled(I, beta, theta, u_r, u0, BN=32, D=8, num_warps=1)
            C = None
        else:
            C = _pick_C(T)
            U, S = scan_forward_v3(I, beta, theta, u_r, u0, C=C, K=K, repair=repair)
        ctx.save_for_backward(I, u0, U, S)
        ctx.cfg = (beta, theta, u_r, alpha, mode, C)
        return S

    @staticmethod
    def backward(ctx, gS):
        I, u0, U, S = ctx.saved_tensors
        beta, theta, u_r, alpha, mode, C = ctx.cfg
        gS = gS.contiguous()
        GU = torch.zeros_like(gS)
        if mode == "serial":
            Lam = serial_backward_tiled(I, u0, U, S, gS, GU, beta, theta, u_r, alpha, BN=32, D=4, num_warps=1)
        else:
            Lam = scan_backward_tiled(I, u0, U, S, gS, GU, beta, theta, u_r, alpha, C=C, BNr=32, D=8, nwr=1)
        return Lam, beta * Lam[0], None, None, None, None, None, None, None


def lif_tuned(I, u0, beta=0.9, theta=1.0, u_r=0.0, alpha=4.0, mode="scan", K=16, repair=0):
    return LIFTuned.apply(I, u0, beta, theta, u_r, alpha, mode, K, repair)


# ============================================================================= v3: tight per-chunk lower bounds
# Lower envelope  l_t = min(u_r, beta * l_{t-1} + I_t),  l_0 = u_0,  satisfies l_t <= u_t
# (u_t is either u_r or v_t = beta*u_{t-1} + I_t >= beta*l_{t-1} + I_t).  Maps of the form
# g(u) = min(c, a*u + b) with a > 0 are closed under composition, so the envelope at every
# chunk entry is itself a cheap two-level scan.  Pruning chunk m's map with lo = l_{mC}
# (instead of the global, very loose min(I)/(1-beta)) keeps only breakpoints near the
# states that can actually enter the chunk.

@triton.jit
def env_chunk_kernel(I, EC, EA, EB, N, C, beta, u_r, BN: tl.constexpr, D: tl.constexpr):
    m = tl.program_id(0)
    pid = tl.program_id(1)
    n = pid * BN + tl.arange(0, BN)
    mask = n < N
    rows = tl.arange(0, D)
    m2 = mask[None, :]
    c = tl.full([BN], float("inf"), tl.float32)
    a = tl.full([BN], 1.0, tl.float32)
    b = tl.zeros([BN], dtype=tl.float32)
    for blk in range(0, C // D):
        off = (m * C + blk * D + rows[:, None]) * N + n[None, :]
        xt = tl.load(I + off, mask=m2, other=0.0)
        for j in tl.static_range(D):
            x = tl.sum(tl.where(rows[:, None] == j, xt, 0.0), axis=0)
            c = tl.minimum(u_r, beta * c + x)
            a = beta * a
            b = beta * b + x
    tl.store(EC + m * N + n, c, mask=mask)
    tl.store(EA + m * N + n, a, mask=mask)
    tl.store(EB + m * N + n, b, mask=mask)


@triton.jit
def env_carry_kernel(U0, EC, EA, EB, Lo, N, M, margin, BN: tl.constexpr):
    pid = tl.program_id(0)
    n = pid * BN + tl.arange(0, BN)
    mask = n < N
    l = tl.load(U0 + n, mask=mask, other=0.0)
    for m in range(M):
        tl.store(Lo + m * N + n, l - margin * (tl.abs(l) + 1.0), mask=mask)
        c = tl.load(EC + m * N + n, mask=mask, other=0.0)
        a = tl.load(EA + m * N + n, mask=mask, other=1.0)
        b = tl.load(EB + m * N + n, mask=mask, other=0.0)
        l = tl.minimum(c, a * l + b)


@triton.jit
def build3_kernel(I, Lo, U0, MA, MB, MX, MC, MN, OVF, N, C, beta, theta, u_r,
                  K: tl.constexpr, BN: tl.constexpr):
    m = tl.program_id(0)
    pid = tl.program_id(1)
    n = pid * BN + tl.arange(0, BN)
    mask = n < N
    ks = tl.arange(0, K)
    lo = tl.load(Lo + m * N + n, mask=mask, other=0.0)
    u0 = tl.load(U0 + n, mask=mask, other=0.0)
    hi = tl.maximum(u0, tl.maximum(theta, u_r))
    A = tl.full([BN], 1.0, tl.float32)
    B = tl.zeros([BN], dtype=tl.float32)
    xd = tl.full([BN, K], float("inf"), tl.float32)
    cd = tl.zeros([BN, K], dtype=tl.float32)
    cnt = tl.zeros([BN], dtype=tl.int32)
    xmin = tl.full([BN], float("inf"), tl.float32)
    ovf = tl.zeros([BN], dtype=tl.int32)
    for j in range(C):
        t = m * C + j
        x = tl.load(I + t * N + n, mask=mask, other=0.0)
        vv = beta * cd + x[:, None]
        cd = tl.where(vv >= theta, u_r, vv)
        num = theta - x - beta * B
        den = beta * A
        z = tl.where(den == 0.0, tl.where(num > 0.0, float("inf"), -float("inf")), num / den)
        new = (z < xmin) & (z <= hi) & (xmin > lo)
        ovf = tl.where(new & (cnt >= K), 1, ovf)
        new = new & (cnt < K)
        slot = (ks[None, :] == cnt[:, None]) & new[:, None]
        xd = tl.where(slot, z[:, None], xd)
        cd = tl.where(slot, u_r, cd)
        cnt = cnt + new.to(tl.int32)
        xmin = tl.where(new, z, xmin)
        A = beta * A
        B = beta * B + x
    bo = m * N + n
    tl.store(MA + bo, A, mask=mask)
    tl.store(MB + bo, B, mask=mask)
    tl.store(MN + bo, cnt, mask=mask)
    tl.store(OVF + bo, ovf, mask=mask)
    off = (m * K + ks[None, :]) * N + n[:, None]
    tl.store(MX + off, xd, mask=mask[:, None])
    tl.store(MC + off, cd, mask=mask[:, None])


def scan_forward_v3(I, beta, theta, u_r, u0, C=64, K=16, BN=32, nw=1, BNc=16, nwc=1,
                    BNr=32, D=8, nwr=1, repair=0, stats=None):
    T, N = I.shape
    M = T // C
    dev = I.device
    EC = torch.empty(M, N, device=dev); EA = torch.empty(M, N, device=dev); EB = torch.empty(M, N, device=dev)
    De = D if C % D == 0 else 1
    env_chunk_kernel[(M, triton.cdiv(N, BNr))](I, EC, EA, EB, N, C, beta, u_r, BN=BNr, D=De, num_warps=nwr)
    Lo = torch.empty(M, N, device=dev)
    env_carry_kernel[(triton.cdiv(N, BNc),)](u0, EC, EA, EB, Lo, N, M, 1e-5, BN=BNc, num_warps=nwc)
    MA = torch.empty(M, N, device=dev); MB = torch.empty(M, N, device=dev)
    MX = torch.empty(M, K, N, device=dev); MC = torch.empty(M, K, N, device=dev)
    MN = torch.empty(M, N, device=dev, dtype=torch.int32)
    OVF = torch.empty(M, N, device=dev, dtype=torch.int32)
    build3_kernel[(M, triton.cdiv(N, BN))](I, Lo, u0, MA, MB, MX, MC, MN, OVF, N, C, beta, theta, u_r,
                                           K=K, BN=BN, num_warps=nw)
    Starts = torch.empty(M, N, device=dev)
    carry2_kernel[(triton.cdiv(N, BNc),)](I, u0, MA, MB, MX, MC, MN, OVF, Starts, N, M, C,
                                          beta, theta, u_r, K=K, BN=BNc, STAGES=1, num_warps=nwc)
    U = torch.empty_like(I); S = torch.empty_like(I); Ends = torch.empty(M, N, device=dev)
    gr = (M, triton.cdiv(N, BNr))
    fwd_tiled_kernel[gr](I, Starts, U, S, Ends, N, C, beta, theta, u_r, BN=BNr, D=D, num_warps=nwr)
    it = 0
    for it in range(1, repair + 1):
        new = torch.cat([u0[None], Ends[:-1]], 0)
        if torch.equal(new, Starts):
            it -= 1
            break
        Starts = new.contiguous()
        fwd_tiled_kernel[gr](I, Starts, U, S, Ends, N, C, beta, theta, u_r, BN=BNr, D=D, num_warps=nwr)
    if stats is not None:
        stats["repair_iters"] = it
        stats["overflow_frac"] = OVF.float().mean().item()
        stats["mean_pieces"] = MN.float().mean().item() + 1
        stats["max_pieces"] = int(MN.max().item()) + 1
    return U, S
