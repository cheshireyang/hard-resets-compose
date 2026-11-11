"""Exact two-level scan for SOFT-reset LIF neurons (subtractive reset u <- v - theta).

Segment maps of soft-reset neurons have a common slope A = beta^L and piecewise-constant offsets,
with at most 1 + L(L+1)/2 pieces (Theorem 7).  Pruned to the reachable interval [l, h] they are
small.  Pieces are stored unsorted with explicit [xl, xr) and offset c (value A*u + c); every step
splits pieces at the threshold preimage and compacts the list with a prefix sum (O(K^2) work per
step and neuron).
  l_t = min(0, beta*l + I_t)            (lower envelope, same algebra as the hard reset)
  h_t = max(theta, beta*h + I_t - theta) (upper envelope, max-affine maps compose too)
Backward: with a detached reset, du_t/dv_t = 1, so the adjoint is lam_t = b_t + beta*lam_{t+1},
a constant-coefficient linear recurrence, solved here by FFT convolution in float64.
"""
import math
import torch
import triton
import triton.language as tl
from . import triton_lif as L


@triton.jit
def soft_env_hi_chunk_kernel(I, EC, EA, EB, N, C, beta, theta, BN: tl.constexpr):
    # compose g(u) = max(c, a*u + b) over the chunk; step: max(theta, beta*u + x - theta)
    m = tl.program_id(0); pid = tl.program_id(1)
    n = pid * BN + tl.arange(0, BN); mask = n < N
    c = tl.full([BN], -float("inf"), tl.float32); a = tl.full([BN], 1.0, tl.float32); b = tl.zeros([BN], dtype=tl.float32)
    for j in range(C):
        x = tl.load(I + (m * C + j) * N + n, mask=mask, other=0.0)
        c = tl.maximum(theta, beta * c + x - theta)
        a = beta * a
        b = beta * b + x - theta
    tl.store(EC + m * N + n, c, mask=mask); tl.store(EA + m * N + n, a, mask=mask); tl.store(EB + m * N + n, b, mask=mask)


@triton.jit
def soft_env_hi_carry_kernel(U0, EC, EA, EB, Hi, N, M, margin, BN: tl.constexpr):
    pid = tl.program_id(0); n = pid * BN + tl.arange(0, BN); mask = n < N
    h = tl.load(U0 + n, mask=mask, other=0.0)
    for m in range(M):
        tl.store(Hi + m * N + n, h + margin * (tl.abs(h) + 1.0), mask=mask)
        c = tl.load(EC + m * N + n, mask=mask, other=0.0)
        a = tl.load(EA + m * N + n, mask=mask, other=1.0)
        b = tl.load(EB + m * N + n, mask=mask, other=0.0)
        h = tl.maximum(c, tl.maximum(a * h + b, h * 0.0 + c))


@triton.jit
def soft_build_kernel(I, Lo, Hi, MA, MXL, MXR, MC, MN, OVF, N, C, beta, theta,
                      K: tl.constexpr, BN: tl.constexpr, LOGK: tl.constexpr):
    m = tl.program_id(0); pid = tl.program_id(1)
    n = pid * BN + tl.arange(0, BN); mask = n < N
    ks = tl.arange(0, K)
    lo = tl.load(Lo + m * N + n, mask=mask, other=0.0)
    hi = tl.load(Hi + m * N + n, mask=mask, other=0.0)
    xl = tl.where(ks[None, :] == 0, -float("inf"), float("inf")) + tl.zeros([BN, K], dtype=tl.float32)
    xr = tl.full([BN, K], float("inf"), tl.float32)
    cc = tl.zeros([BN, K], dtype=tl.float32)
    cnt = tl.full([BN], 1, tl.int32)
    ovf = tl.zeros([BN], dtype=tl.int32)
    A = tl.full([BN], 1.0, tl.float32)
    for j in range(C):
        x = tl.load(I + (m * C + j) * N + n, mask=mask, other=0.0)
        ct = (theta - x) / beta
        valid = ks[None, :] < cnt[:, None]
        num = ct[:, None] - cc
        z = tl.where(A[:, None] == 0.0, tl.where(num > 0.0, float("inf"), -float("inf")), num / A[:, None])
        full = z <= xl
        split = (z > xl) & (z < xr) & valid
        cns = beta * cc + x[:, None]
        cl = tl.where(full, cns - theta, cns)
        xr_left = tl.where(split, z, xr)
        vl = valid & (xr_left > lo[:, None]) & (xl <= hi[:, None])
        vr = split & (xr > lo[:, None]) & (z <= hi[:, None])
        # O(K log K) compaction: piece i emits vl_i + vr_i outputs starting at exclusive prefix p_i;
        # target slot k is owned by the largest i with p_i <= k (binary search with gathers)
        nout = vl.to(tl.int32) + vr.to(tl.int32)
        p = tl.cumsum(nout, axis=1) - nout
        ntot = tl.sum(nout, axis=1)
        ovf = tl.where(ntot > K, 1, ovf)
        kk = ks[None, :] + tl.zeros([BN, K], dtype=tl.int32)
        lo_i = tl.zeros([BN, K], dtype=tl.int32)
        hi_i = tl.full([BN, K], K - 1, tl.int32)
        for _ in tl.static_range(LOGK):
            mid = (lo_i + hi_i + 1) // 2
            pm = tl.gather(p, mid, axis=1)
            le = pm <= kk
            lo_i = tl.where(le, mid, lo_i)
            hi_i = tl.where(le, hi_i, mid - 1)
        idx = lo_i
        d = kk - tl.gather(p, idx, axis=1)
        vl_g = tl.gather(vl.to(tl.int32), idx, axis=1) > 0
        is_left = (d == 0) & vl_g
        cr = cns - theta
        nxl = tl.where(is_left, tl.gather(xl, idx, axis=1), tl.gather(z, idx, axis=1))
        nxr = tl.where(is_left, tl.gather(xr_left, idx, axis=1), tl.gather(xr, idx, axis=1))
        nc = tl.where(is_left, tl.gather(cl, idx, axis=1), tl.gather(cr, idx, axis=1))
        cnt = tl.minimum(ntot, K)
        keep = ks[None, :] < cnt[:, None]
        xl = tl.where(keep, nxl, float("inf")); xr = tl.where(keep, nxr, float("inf")); cc = tl.where(keep, nc, 0.0)
        A = beta * A
    bo = m * N + n
    tl.store(MA + bo, A, mask=mask); tl.store(MN + bo, cnt, mask=mask); tl.store(OVF + bo, ovf, mask=mask)
    off = (m * K + ks[None, :]) * N + n[:, None]
    tl.store(MXL + off, xl, mask=mask[:, None]); tl.store(MXR + off, xr, mask=mask[:, None]); tl.store(MC + off, cc, mask=mask[:, None])


@triton.jit
def _soft_step(u, x, beta, theta):
    v = beta * u + x
    s = v >= theta
    return tl.where(s, v - theta, v), s


@triton.jit
def soft_carry_kernel(I, U0, MA, MXL, MXR, MC, MN, OVF, Starts, N, M, C, beta, theta,
                      K: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0); n = pid * BN + tl.arange(0, BN); mask = n < N
    ks = tl.arange(0, K)
    u = tl.load(U0 + n, mask=mask, other=0.0)
    for m in range(M):
        tl.store(Starts + m * N + n, u, mask=mask)
        bo = m * N + n
        A = tl.load(MA + bo, mask=mask, other=1.0)
        cnt = tl.load(MN + bo, mask=mask, other=0)
        ovf = tl.load(OVF + bo, mask=mask, other=0)
        off = (m * K + ks[None, :]) * N + n[:, None]
        xl = tl.load(MXL + off, mask=mask[:, None], other=float("inf"))
        xr = tl.load(MXR + off, mask=mask[:, None], other=float("inf"))
        cc = tl.load(MC + off, mask=mask[:, None], other=0.0)
        hit = (ks[None, :] < cnt[:, None]) & (xl <= u[:, None]) & (u[:, None] < xr)
        c = tl.sum(tl.where(hit, cc, 0.0), axis=1)
        found = tl.sum(hit.to(tl.int32), axis=1) == 1
        nu = A * u + c
        bad = ((ovf > 0) | (~found)) & mask
        if tl.max(bad.to(tl.int32), axis=0) > 0:   # overflow or state outside the pruned domain: serial
            w = u
            for j in range(C):
                x = tl.load(I + (m * C + j) * N + n, mask=mask, other=0.0)
                w, s_ = _soft_step(w, x, beta, theta)
            nu = tl.where(bad, w, nu)
        u = nu


@triton.jit
def soft_replay_kernel(I, Starts, Uout, Sout, Ends, N, C, beta, theta, BN: tl.constexpr):
    m = tl.program_id(0); pid = tl.program_id(1)
    n = pid * BN + tl.arange(0, BN); mask = n < N
    u = tl.load(Starts + m * N + n, mask=mask, other=0.0)
    for j in range(C):
        t = m * C + j
        x = tl.load(I + t * N + n, mask=mask, other=0.0)
        u, s = _soft_step(u, x, beta, theta)
        tl.store(Uout + t * N + n, u, mask=mask)
        tl.store(Sout + t * N + n, s.to(tl.float32), mask=mask)
    tl.store(Ends + m * N + n, u, mask=mask)


@triton.jit
def soft_serial_kernel(I, U0, Uout, Sout, T, N, beta, theta, BN: tl.constexpr):
    pid = tl.program_id(0); n = pid * BN + tl.arange(0, BN); mask = n < N
    u = tl.load(U0 + n, mask=mask, other=0.0)
    for t in range(T):
        x = tl.load(I + t * N + n, mask=mask, other=0.0)
        u, s = _soft_step(u, x, beta, theta)
        tl.store(Uout + t * N + n, u, mask=mask)
        tl.store(Sout + t * N + n, s.to(tl.float32), mask=mask)


def soft_serial_forward(I, beta, theta, u0, BN=32):
    T, N = I.shape
    U = torch.empty_like(I); S = torch.empty_like(I)
    soft_serial_kernel[(triton.cdiv(N, BN),)](I, u0, U, S, T, N, beta, theta, BN=BN, num_warps=1)
    return U, S


def soft_scan_forward(I, beta, theta, u0, C=32, K=16, BN=16, nw=1, BNc=16, BNr=32, repair=0, stats=None):
    T, N = I.shape
    assert T % C == 0
    M = T // C
    dev = I.device
    # envelopes: lower (hard-reset algebra with u_r = 0) and upper (max-affine)
    EC = torch.empty(M, N, device=dev); EA = torch.empty(M, N, device=dev); EB = torch.empty(M, N, device=dev)
    L.env_chunk_kernel[(M, triton.cdiv(N, BNr))](I, EC, EA, EB, N, C, beta, 0.0, BN=BNr, D=1, num_warps=1)
    Lo = torch.empty(M, N, device=dev)
    L.env_carry_kernel[(triton.cdiv(N, BNc),)](u0, EC, EA, EB, Lo, N, M, 1e-5, BN=BNc, num_warps=1)
    soft_env_hi_chunk_kernel[(M, triton.cdiv(N, BNr))](I, EC, EA, EB, N, C, beta, theta, BN=BNr, num_warps=1)
    Hi = torch.empty(M, N, device=dev)
    soft_env_hi_carry_kernel[(triton.cdiv(N, BNc),)](u0, EC, EA, EB, Hi, N, M, 1e-5, BN=BNc, num_warps=1)
    MA = torch.empty(M, N, device=dev)
    MXL = torch.empty(M, K, N, device=dev); MXR = torch.empty(M, K, N, device=dev); MC = torch.empty(M, K, N, device=dev)
    MN = torch.empty(M, N, device=dev, dtype=torch.int32); OVF = torch.empty(M, N, device=dev, dtype=torch.int32)
    soft_build_kernel[(M, triton.cdiv(N, BN))](I, Lo, Hi, MA, MXL, MXR, MC, MN, OVF, N, C, beta, theta,
                                               K=K, BN=BN, LOGK=int(math.log2(K)), num_warps=nw)
    Starts = torch.empty(M, N, device=dev)
    soft_carry_kernel[(triton.cdiv(N, BNc),)](I, u0, MA, MXL, MXR, MC, MN, OVF, Starts, N, M, C, beta, theta,
                                              K=K, BN=BNc, num_warps=1)
    U = torch.empty_like(I); S = torch.empty_like(I); Ends = torch.empty(M, N, device=dev)
    gr = (M, triton.cdiv(N, BNr))
    soft_replay_kernel[gr](I, Starts, U, S, Ends, N, C, beta, theta, BN=BNr, num_warps=1)
    it = 0
    for it in range(1, repair + 1):
        new = torch.cat([u0[None], Ends[:-1]], 0)
        if torch.equal(new, Starts):
            it -= 1; break
        Starts = new.contiguous()
        soft_replay_kernel[gr](I, Starts, U, S, Ends, N, C, beta, theta, BN=BNr, num_warps=1)
    if stats is not None:
        stats["repair_iters"] = it
        stats["overflow_frac"] = OVF.float().mean().item()
        stats["mean_pieces"] = MN.float().mean().item()
        stats["max_pieces"] = int(MN.max().item())
    return U, S


def soft_adjoint_fft(V, gS, beta, theta, alpha=4.0):
    """lam_t = b_t + beta*lam_{t+1}, b_t = gS_t * sigma'(V_t - theta)  (detached soft reset)."""
    T = V.shape[0]
    sg = torch.sigmoid(alpha * (V - theta)); b = (gS * alpha * sg * (1 - sg)).double()
    k = beta ** torch.arange(T, device=V.device, dtype=torch.float64)
    n = 2 * T
    lam = torch.fft.irfft(torch.fft.rfft(b.flip(0), n=n, dim=0) * torch.fft.rfft(k, n=n)[:, None], n=n, dim=0)[:T].flip(0)
    return lam.to(V.dtype)


@triton.jit
def soft_fwd_tiled_kernel(I, Starts, Uout, Sout, Ends, N, Ls, beta, theta, BN: tl.constexpr, D: tl.constexpr):
    """Time-tiled soft-reset recurrence over chunk m (serial baseline: M = 1, Ls = T)."""
    m = tl.program_id(0); pid = tl.program_id(1)
    n = pid * BN + tl.arange(0, BN); mask = n < N
    rows = tl.arange(0, D)
    u = tl.load(Starts + m * N + n, mask=mask, other=0.0)
    base = m * Ls * N
    off = rows[:, None] * N + n[None, :]
    m2 = mask[None, :]
    nt = Ls // D
    xt = tl.load(I + base + off, mask=m2, other=0.0)
    for b in range(0, nt):
        xn = tl.load(I + base + (b + 1) * D * N + off, mask=m2 & (b + 1 < nt), other=0.0)
        ut = tl.zeros([D, BN], dtype=tl.float32); st = tl.zeros([D, BN], dtype=tl.float32)
        for j in tl.static_range(D):
            x = tl.sum(tl.where(rows[:, None] == j, xt, 0.0), axis=0)
            u, s = _soft_step(u, x, beta, theta)
            ut = tl.where(rows[:, None] == j, u[None, :], ut)
            st = tl.where(rows[:, None] == j, s.to(tl.float32)[None, :], st)
        tl.store(Uout + base + b * D * N + off, ut, mask=m2)
        tl.store(Sout + base + b * D * N + off, st, mask=m2)
        xt = xn
    tl.store(Ends + m * N + n, u, mask=mask)


def soft_serial_forward_tiled(I, beta, theta, u0, BN=32, D=8, num_warps=1):
    T, N = I.shape
    U = torch.empty_like(I); S = torch.empty_like(I); E = torch.empty(1, N, device=I.device)
    soft_fwd_tiled_kernel[(1, triton.cdiv(N, BN))](I, u0, U, S, E, N, T, beta, theta, BN=BN, D=D, num_warps=num_warps)
    return U, S
