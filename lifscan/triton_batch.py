"""Serial LIF and ALIF kernels with per-recurrence parameters, for fitting many cells in one batched call.

Every column n of I is an independent recurrence with its own beta[n], u_r[n] (LIF) or beta, rho, q, u_r (ALIF).  The
kernels are the time-tiled, prefetching serial kernels of triton_lif / triton_alif with the scalar parameters replaced
by per-column loads; the step arithmetic is the same (shared @triton.jit step functions), so a column computes
bit-identically to a scalar-parameter call with that column's values.  Gradients are returned per column."""
import torch
import triton
import triton.language as tl
from .triton_lif import _step, _coef, serial_fwd_cfg, serial_bwd_cfg
from .triton_alif import _alif_step, _alif_sg, alif_fwd_cfg


# ----------------------------------------------------------------------------- LIF
@triton.jit
def lif_vec_fwd_kernel(I, U0, Beta, Ur, Uout, Sout, N, T, theta, BN: tl.constexpr, D: tl.constexpr):
    pid = tl.program_id(0)
    n = pid * BN + tl.arange(0, BN)
    mask = n < N
    rows = tl.arange(0, D)
    u = tl.load(U0 + n, mask=mask, other=0.0)
    beta = tl.load(Beta + n, mask=mask, other=0.0)
    u_r = tl.load(Ur + n, mask=mask, other=0.0)
    off = rows[:, None] * N + n[None, :]
    m2 = mask[None, :]
    nt = T // D
    xt = tl.load(I + off, mask=m2, other=0.0)
    for b in range(0, nt):
        xn = tl.load(I + (b + 1) * D * N + off, mask=m2 & (b + 1 < nt), other=0.0)
        ut = tl.zeros([D, BN], dtype=tl.float32)
        st = tl.zeros([D, BN], dtype=tl.float32)
        for j in tl.static_range(D):
            x = tl.sum(tl.where(rows[:, None] == j, xt, 0.0), axis=0)
            u, s, v = _step(u, x, beta, theta, u_r)
            ut = tl.where(rows[:, None] == j, u[None, :], ut)
            st = tl.where(rows[:, None] == j, s.to(tl.float32)[None, :], st)
        tl.store(Uout + b * D * N + off, ut, mask=m2)
        tl.store(Sout + b * D * N + off, st, mask=m2)
        xt = xn


@triton.jit
def lif_vec_bwd_kernel(I, U0, Uout, Sout, GS, GU, Lam, Beta, Ur, N, T, theta, alpha,
                       DETACH: tl.constexpr, BN: tl.constexpr, D: tl.constexpr):
    pid = tl.program_id(0)
    n = pid * BN + tl.arange(0, BN)
    mask = n < N
    rows = tl.arange(0, D)
    m2 = mask[None, :]
    lam = tl.zeros([BN], dtype=tl.float32)
    u0 = tl.load(U0 + n, mask=mask, other=0.0)
    beta = tl.load(Beta + n, mask=mask, other=0.0)
    u_r = tl.load(Ur + n, mask=mask, other=0.0)
    nt = T // D
    t0 = (nt - 1) * D
    off = (t0 + rows[:, None]) * N + n[None, :]
    xt = tl.load(I + off, mask=m2, other=0.0); stl = tl.load(Sout + off, mask=m2, other=0.0)
    gst = tl.load(GS + off, mask=m2, other=0.0); gut = tl.load(GU + off, mask=m2, other=0.0)
    upt = tl.load(Uout + off - N, mask=m2 & (t0 + rows[:, None] > 0), other=0.0)
    for i in range(0, nt):
        b = nt - 1 - i
        t0 = b * D
        offn = (t0 - D + rows[:, None]) * N + n[None, :]
        mn = m2 & (b >= 1)
        xn = tl.load(I + offn, mask=mn, other=0.0); sn = tl.load(Sout + offn, mask=mn, other=0.0)
        gsn = tl.load(GS + offn, mask=mn, other=0.0); gun = tl.load(GU + offn, mask=mn, other=0.0)
        upn = tl.load(Uout + offn - N, mask=mn & (t0 - D + rows[:, None] > 0), other=0.0)
        upc = tl.where(t0 + rows[:, None] == 0, u0[None, :], upt)
        lt = tl.zeros([D, BN], dtype=tl.float32)
        for jj in tl.static_range(D):
            j = D - 1 - jj
            sel = rows[:, None] == j
            x = tl.sum(tl.where(sel, xt, 0.0), axis=0)
            s = tl.sum(tl.where(sel, stl, 0.0), axis=0)
            gs = tl.sum(tl.where(sel, gst, 0.0), axis=0)
            gu = tl.sum(tl.where(sel, gut, 0.0), axis=0)
            up = tl.sum(tl.where(sel, upc, 0.0), axis=0)
            a, bb = _coef(up, s, x, gs, gu, beta, theta, u_r, alpha, DETACH)
            lam = bb + a * lam
            lt = tl.where(sel, lam[None, :], lt)
        tl.store(Lam + (t0 + rows[:, None]) * N + n[None, :], lt, mask=m2)
        xt = xn; stl = sn; gst = gsn; gut = gun; upt = upn


def _cfg(fn, N, T):
    BN, D, nw = fn(N)
    while T % D: D //= 2
    return BN, D, nw


def lif_vec_forward(I, beta, u_r, u0, theta=1.0, cfg=None):
    T, N = I.shape
    BN, D, nw = cfg or _cfg(serial_fwd_cfg, N, T)
    U = torch.empty_like(I); S = torch.empty_like(I)
    lif_vec_fwd_kernel[(triton.cdiv(N, BN),)](I, u0, beta, u_r, U, S, N, T, theta, BN=BN, D=D, num_warps=nw)
    return U, S


def lif_vec_backward(I, u0, U, S, GS, beta, u_r, theta=1.0, alpha=4.0, detach=True, cfg=None):
    T, N = I.shape
    BN, D, nw = cfg or _cfg(serial_bwd_cfg, N, T)
    Lam = torch.empty_like(I); GU = torch.zeros_like(I)
    lif_vec_bwd_kernel[(triton.cdiv(N, BN),)](I, u0, U, S, GS, GU, Lam, beta, u_r, N, T, theta, alpha,
                                              DETACH=detach, BN=BN, D=D, num_warps=nw)
    return Lam


class LIFVecFn(torch.autograd.Function):
    """S = LIF(I; beta[n], u_r[n]), theta fixed; gradients for I, beta (per column) and u_r (per column)."""

    @staticmethod
    def forward(ctx, I, u0, beta, u_r, theta, alpha):
        I = I.contiguous(); b = beta.detach().contiguous().float(); ur = u_r.detach().contiguous().float()
        U, S = lif_vec_forward(I, b, ur, u0, theta)
        ctx.save_for_backward(I, u0, U, S, b, ur)
        ctx.c = (theta, alpha)
        return S

    @staticmethod
    def backward(ctx, gS):
        I, u0, U, S, b, ur = ctx.saved_tensors
        theta, alpha = ctx.c
        lam = lif_vec_backward(I, u0, U, S, gS.contiguous(), b, ur, theta, alpha)
        uprev = torch.cat([u0[None], U[:-1]], 0)
        g_beta = (lam * uprev).sum(0)
        lam_next = torch.cat([lam[1:], torch.zeros_like(lam[:1])], 0)
        g_ur = (S * b * lam_next).sum(0)
        return lam, b * lam[0], g_beta, g_ur, None, None


def lif_vec(I, u0, beta, u_r, theta=1.0, alpha=4.0):
    return LIFVecFn.apply(I, u0, beta, u_r, theta, alpha)


# ----------------------------------------------------------------------------- ALIF
@triton.jit
def alif_vec_fwd_kernel(I, U0, W0, Beta, Rho, Q, Ur, Uout, Wout, Sout, N, T, theta0, BN: tl.constexpr, D: tl.constexpr):
    pid = tl.program_id(0)
    n = pid * BN + tl.arange(0, BN); mask = n < N
    rows = tl.arange(0, D)
    u = tl.load(U0 + n, mask=mask, other=0.0); w = tl.load(W0 + n, mask=mask, other=0.0)
    beta = tl.load(Beta + n, mask=mask, other=0.0); rho = tl.load(Rho + n, mask=mask, other=0.0)
    q = tl.load(Q + n, mask=mask, other=0.0); u_r = tl.load(Ur + n, mask=mask, other=0.0)
    off = rows[:, None] * N + n[None, :]
    m2 = mask[None, :]
    nt = T // D
    xt = tl.load(I + off, mask=m2, other=0.0)
    for b in range(0, nt):
        xn = tl.load(I + (b + 1) * D * N + off, mask=m2 & (b + 1 < nt), other=0.0)
        ut = tl.zeros([D, BN], dtype=tl.float32); wt = tl.zeros([D, BN], dtype=tl.float32); st = tl.zeros([D, BN], dtype=tl.float32)
        for j in tl.static_range(D):
            x = tl.sum(tl.where(rows[:, None] == j, xt, 0.0), axis=0)
            u, w, s = _alif_step(u, w, x, beta, rho, q, theta0, u_r)
            ut = tl.where(rows[:, None] == j, u[None, :], ut)
            wt = tl.where(rows[:, None] == j, w[None, :], wt)
            st = tl.where(rows[:, None] == j, s.to(tl.float32)[None, :], st)
        tl.store(Uout + b * D * N + off, ut, mask=m2)
        tl.store(Wout + b * D * N + off, wt, mask=m2)
        tl.store(Sout + b * D * N + off, st, mask=m2)
        xt = xn


@triton.jit
def alif_vec_bwd_kernel(I, U0, W0, Uout, Wout, Sout, GS, LW, LV, Beta, Rho, Q, N, T, theta0, alpha,
                        BN: tl.constexpr, D: tl.constexpr):
    pid = tl.program_id(0)
    n = pid * BN + tl.arange(0, BN); mask = n < N
    rows = tl.arange(0, D); m2 = mask[None, :]
    u0 = tl.load(U0 + n, mask=mask, other=0.0); w0 = tl.load(W0 + n, mask=mask, other=0.0)
    beta = tl.load(Beta + n, mask=mask, other=0.0); rho = tl.load(Rho + n, mask=mask, other=0.0)
    q = tl.load(Q + n, mask=mask, other=0.0)
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


def alif_vec_forward(I, beta, rho, q, u_r, u0, w0, theta0=1.0, cfg=None):
    T, N = I.shape
    BN, D, nw = cfg or _cfg(alif_fwd_cfg, N, T)
    U = torch.empty_like(I); W = torch.empty_like(I); S = torch.empty_like(I)
    alif_vec_fwd_kernel[(triton.cdiv(N, BN),)](I, u0, w0, beta, rho, q, u_r, U, W, S, N, T, theta0, BN=BN, D=D, num_warps=nw)
    return U, W, S


def alif_vec_backward(I, U, W, S, GS, beta, rho, q, u0, w0, theta0=1.0, alpha=4.0, cfg=None):
    T, N = I.shape
    BN, D, nw = cfg or _cfg(serial_bwd_cfg, N, T)
    LW = torch.empty_like(I); LV = torch.empty_like(I)
    alif_vec_bwd_kernel[(triton.cdiv(N, BN),)](I, u0, w0, U, W, S, GS, LW, LV, beta, rho, q, N, T, theta0, alpha,
                                               BN=BN, D=D, num_warps=nw)
    return LV, LW


class ALIFVecFn(torch.autograd.Function):
    """S = ALIF(I; beta, rho, q, u_r per column), theta0 fixed, detached reset; gradients for I, beta, rho, q."""

    @staticmethod
    def forward(ctx, I, u0, w0, beta, rho, q, u_r, theta0, alpha):
        I = I.contiguous()
        b, r, qq, ur = (t.detach().contiguous().float() for t in (beta, rho, q, u_r))
        U, W, S = alif_vec_forward(I, b, r, qq, ur, u0, w0, theta0)
        ctx.save_for_backward(I, u0, w0, U, W, S, b, r, qq)
        ctx.c = (theta0, alpha)
        ctx.mark_non_differentiable(U, W)
        return S, U, W

    @staticmethod
    def backward(ctx, gS, gU, gW):
        I, u0, w0, U, W, S, b, r, qq = ctx.saved_tensors
        th, alpha = ctx.c
        GS = gS.contiguous()
        LV, LW = alif_vec_backward(I, U, W, S, GS, b, r, qq, u0, w0, th, alpha)
        Up = torch.cat([u0[None], U[:-1]], 0); Wp = torch.cat([w0[None], W[:-1]], 0)
        return (LV, b * LV[0], None, (LV * Up).sum(0), (LW * Wp).sum(0), (LW * S).sum(0), None, None, None)


def alif_vec(I, u0, w0, beta, rho, q, u_r, theta0=1.0, alpha=4.0):
    return ALIFVecFn.apply(I, u0, w0, beta, rho, q, u_r, theta0, alpha)
