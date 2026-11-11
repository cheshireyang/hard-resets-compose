"""Exact two-level scan for LIF neurons with an AFFINE reset  u <- r(v) = gamma*v + delta,
0 <= gamma <= 1.  This family contains the hard reset (gamma = 0, delta = u_r) and the soft reset
(gamma = 1, delta = -theta); by Theorem 7(b) it is order-preserving and non-expanding, so its
segment maps have at most 1 + L(L+1)/2 pieces and the recurrence is a scan.

Pieces are stored unsorted with explicit [xl, xr), slope a and offset c (value a*u + c).  Per step:
no spike: (a, c) -> (beta*a, beta*c + x);   spike: (a, c) -> (gamma*beta*a, gamma*(beta*c + x) + delta).
Reachable interval:  l_t = min(r(theta), beta*l + I_t)   (min-affine scan, = hard envelope with u_r = r(theta))
                     h_t = max(theta, gamma*(beta*h + I_t) + delta)   (max-affine scan)
Compaction: prefix sums + binary search + gather, O(K log K) per step.
"""
import math
import torch
import triton
import triton.language as tl
from . import triton_lif as L


@triton.jit
def _aff_step(u, x, beta, theta, gamma, delta):
    v = beta * u + x
    s = v >= theta
    return tl.where(s, gamma * v + delta, v), s, v


@triton.jit
def aff_env_hi_chunk_kernel(I, EC, EA, EB, N, C, beta, theta, gamma, delta, BN: tl.constexpr):
    m = tl.program_id(0); pid = tl.program_id(1)
    n = pid * BN + tl.arange(0, BN); mask = n < N
    c = tl.full([BN], -float("inf"), tl.float32); a = tl.full([BN], 1.0, tl.float32); b = tl.zeros([BN], dtype=tl.float32)
    for j in range(C):
        x = tl.load(I + (m * C + j) * N + n, mask=mask, other=0.0)
        c = tl.maximum(theta, gamma * (beta * c + x) + delta)
        a = gamma * beta * a
        b = gamma * (beta * b + x) + delta
    tl.store(EC + m * N + n, c, mask=mask); tl.store(EA + m * N + n, a, mask=mask); tl.store(EB + m * N + n, b, mask=mask)


@triton.jit
def aff_env_hi_carry_kernel(U0, EC, EA, EB, Hi, N, M, margin, BN: tl.constexpr):
    pid = tl.program_id(0); n = pid * BN + tl.arange(0, BN); mask = n < N
    h = tl.load(U0 + n, mask=mask, other=0.0)
    for m in range(M):
        tl.store(Hi + m * N + n, h + margin * (tl.abs(h) + 1.0), mask=mask)
        c = tl.load(EC + m * N + n, mask=mask, other=0.0)
        a = tl.load(EA + m * N + n, mask=mask, other=1.0)
        b = tl.load(EB + m * N + n, mask=mask, other=0.0)
        h = tl.maximum(c, a * h + b)


@triton.jit
def aff_build_kernel(I, Lo, Hi, MXL, MXR, MA, MC, MN, OVF, N, C, beta, theta, gamma, delta,
                     K: tl.constexpr, BN: tl.constexpr, LOGK: tl.constexpr):
    m = tl.program_id(0); pid = tl.program_id(1)
    n = pid * BN + tl.arange(0, BN); mask = n < N
    ks = tl.arange(0, K)
    lo = tl.load(Lo + m * N + n, mask=mask, other=0.0)
    hi = tl.load(Hi + m * N + n, mask=mask, other=0.0)
    xl = tl.where(ks[None, :] == 0, -float("inf"), float("inf")) + tl.zeros([BN, K], dtype=tl.float32)
    xr = tl.full([BN, K], float("inf"), tl.float32)
    aa = tl.full([BN, K], 1.0, tl.float32)
    cc = tl.zeros([BN, K], dtype=tl.float32)
    cnt = tl.full([BN], 1, tl.int32)
    ovf = tl.zeros([BN], dtype=tl.int32)
    for j in range(C):
        x = tl.load(I + (m * C + j) * N + n, mask=mask, other=0.0)
        ct = (theta - x) / beta
        valid = ks[None, :] < cnt[:, None]
        num = ct[:, None] - cc
        # spike on the piece iff a*u + c >= ct  <=>  u >= z (a > 0);  a == 0: all or nothing
        z = tl.where(aa == 0.0, tl.where(num > 0.0, float("inf"), -float("inf")), num / aa)
        full = z <= xl
        split = (z > xl) & (z < xr) & valid
        vns = beta * cc + x[:, None]
        c_no = vns
        c_sp = gamma * vns + delta
        a_no = beta * aa
        a_sp = gamma * beta * aa
        cl = tl.where(full, c_sp, c_no)
        al = tl.where(full, a_sp, a_no)
        xr_left = tl.where(split, z, xr)
        vl = valid & (xr_left > lo[:, None]) & (xl <= hi[:, None])
        vr = split & (xr > lo[:, None]) & (z <= hi[:, None])
        nout = vl.to(tl.int32) + vr.to(tl.int32)
        p = tl.cumsum(nout, axis=1) - nout
        ntot = tl.sum(nout, axis=1)
        ovf = tl.where(ntot > K, 1, ovf)
        kk = ks[None, :] + tl.zeros([BN, K], dtype=tl.int32)
        lo_i = tl.zeros([BN, K], dtype=tl.int32)
        hi_i = tl.full([BN, K], K - 1, tl.int32)
        for _ in tl.static_range(LOGK):
            mid = (lo_i + hi_i + 1) // 2
            le = tl.gather(p, mid, axis=1) <= kk
            lo_i = tl.where(le, mid, lo_i)
            hi_i = tl.where(le, hi_i, mid - 1)
        idx = lo_i
        d = kk - tl.gather(p, idx, axis=1)
        is_left = (d == 0) & (tl.gather(vl.to(tl.int32), idx, axis=1) > 0)
        nxl = tl.where(is_left, tl.gather(xl, idx, axis=1), tl.gather(z, idx, axis=1))
        nxr = tl.where(is_left, tl.gather(xr_left, idx, axis=1), tl.gather(xr, idx, axis=1))
        na = tl.where(is_left, tl.gather(al, idx, axis=1), tl.gather(a_sp, idx, axis=1))
        nc = tl.where(is_left, tl.gather(cl, idx, axis=1), tl.gather(c_sp, idx, axis=1))
        cnt = tl.minimum(ntot, K)
        keep = ks[None, :] < cnt[:, None]
        xl = tl.where(keep, nxl, float("inf")); xr = tl.where(keep, nxr, float("inf"))
        aa = tl.where(keep, na, 1.0); cc = tl.where(keep, nc, 0.0)
    bo = m * N + n
    tl.store(MN + bo, cnt, mask=mask); tl.store(OVF + bo, ovf, mask=mask)
    off = (m * K + ks[None, :]) * N + n[:, None]
    tl.store(MXL + off, xl, mask=mask[:, None]); tl.store(MXR + off, xr, mask=mask[:, None])
    tl.store(MA + off, aa, mask=mask[:, None]); tl.store(MC + off, cc, mask=mask[:, None])


@triton.jit
def aff_carry_kernel(I, U0, MXL, MXR, MA, MC, MN, OVF, Starts, N, M, C, beta, theta, gamma, delta,
                     K: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0); n = pid * BN + tl.arange(0, BN); mask = n < N
    ks = tl.arange(0, K)
    u = tl.load(U0 + n, mask=mask, other=0.0)
    for m in range(M):
        tl.store(Starts + m * N + n, u, mask=mask)
        bo = m * N + n
        cnt = tl.load(MN + bo, mask=mask, other=0)
        ovf = tl.load(OVF + bo, mask=mask, other=0)
        off = (m * K + ks[None, :]) * N + n[:, None]
        xl = tl.load(MXL + off, mask=mask[:, None], other=float("inf"))
        xr = tl.load(MXR + off, mask=mask[:, None], other=float("inf"))
        aa = tl.load(MA + off, mask=mask[:, None], other=0.0)
        cc = tl.load(MC + off, mask=mask[:, None], other=0.0)
        hit = (ks[None, :] < cnt[:, None]) & (xl <= u[:, None]) & (u[:, None] < xr)
        nu = tl.sum(tl.where(hit, aa * u[:, None] + cc, 0.0), axis=1)
        found = tl.sum(hit.to(tl.int32), axis=1) == 1
        bad = ((ovf > 0) | (~found)) & mask
        if tl.max(bad.to(tl.int32), axis=0) > 0:
            w = u
            for j in range(C):
                x = tl.load(I + (m * C + j) * N + n, mask=mask, other=0.0)
                w, s_, v_ = _aff_step(w, x, beta, theta, gamma, delta)
            nu = tl.where(bad, w, nu)
        u = nu


@triton.jit
def aff_run_kernel(I, Starts, Uout, Sout, Ends, N, Ls, beta, theta, gamma, delta, BN: tl.constexpr):
    """Replay chunk m from its start (serial baseline: one chunk of length T)."""
    m = tl.program_id(0); pid = tl.program_id(1)
    n = pid * BN + tl.arange(0, BN); mask = n < N
    u = tl.load(Starts + m * N + n, mask=mask, other=0.0)
    for j in range(Ls):
        t = m * Ls + j
        x = tl.load(I + t * N + n, mask=mask, other=0.0)
        u, s, v = _aff_step(u, x, beta, theta, gamma, delta)
        tl.store(Uout + t * N + n, u, mask=mask)
        tl.store(Sout + t * N + n, s.to(tl.float32), mask=mask)
    tl.store(Ends + m * N + n, u, mask=mask)


def aff_serial_forward(I, beta, theta, gamma, delta, u0, BN=32):
    T, N = I.shape
    U = torch.empty_like(I); S = torch.empty_like(I); E = torch.empty(1, N, device=I.device)
    aff_run_kernel[(1, triton.cdiv(N, BN))](I, u0, U, S, E, N, T, beta, theta, gamma, delta, BN=BN, num_warps=1)
    return U, S


def aff_scan_forward(I, beta, theta, gamma, delta, u0, C=128, K=32, BN=16, BNc=16, BNr=32, repair=0, stats=None):
    T, N = I.shape
    assert T % C == 0 and 0.0 <= gamma <= 1.0
    M = T // C
    dev = I.device
    EC = torch.empty(M, N, device=dev); EA = torch.empty(M, N, device=dev); EB = torch.empty(M, N, device=dev)
    r_theta = gamma * theta + delta
    L.env_chunk_kernel[(M, triton.cdiv(N, BNr))](I, EC, EA, EB, N, C, beta, r_theta, BN=BNr, D=1, num_warps=1)
    Lo = torch.empty(M, N, device=dev)
    L.env_carry_kernel[(triton.cdiv(N, BNc),)](u0, EC, EA, EB, Lo, N, M, 1e-5, BN=BNc, num_warps=1)
    aff_env_hi_chunk_kernel[(M, triton.cdiv(N, BNr))](I, EC, EA, EB, N, C, beta, theta, gamma, delta, BN=BNr, num_warps=1)
    Hi = torch.empty(M, N, device=dev)
    aff_env_hi_carry_kernel[(triton.cdiv(N, BNc),)](u0, EC, EA, EB, Hi, N, M, 1e-5, BN=BNc, num_warps=1)
    MXL = torch.empty(M, K, N, device=dev); MXR = torch.empty_like(MXL); MA = torch.empty_like(MXL); MC = torch.empty_like(MXL)
    MN = torch.empty(M, N, device=dev, dtype=torch.int32); OVF = torch.empty(M, N, device=dev, dtype=torch.int32)
    aff_build_kernel[(M, triton.cdiv(N, BN))](I, Lo, Hi, MXL, MXR, MA, MC, MN, OVF, N, C, beta, theta, gamma, delta,
                                              K=K, BN=BN, LOGK=int(math.log2(K)), num_warps=1)
    Starts = torch.empty(M, N, device=dev)
    aff_carry_kernel[(triton.cdiv(N, BNc),)](I, u0, MXL, MXR, MA, MC, MN, OVF, Starts, N, M, C, beta, theta, gamma, delta,
                                             K=K, BN=BNc, num_warps=1)
    U = torch.empty_like(I); S = torch.empty_like(I); Ends = torch.empty(M, N, device=dev)
    gr = (M, triton.cdiv(N, BNr))
    aff_run_kernel[gr](I, Starts, U, S, Ends, N, C, beta, theta, gamma, delta, BN=BNr, num_warps=1)
    it = 0
    for it in range(1, repair + 1):
        new = torch.cat([u0[None], Ends[:-1]], 0)
        if torch.equal(new, Starts):
            it -= 1; break
        Starts = new.contiguous()
        aff_run_kernel[gr](I, Starts, U, S, Ends, N, C, beta, theta, gamma, delta, BN=BNr, num_warps=1)
    if stats is not None:
        stats["repair_iters"] = it
        stats["overflow_frac"] = OVF.float().mean().item()
        stats["mean_pieces"] = MN.float().mean().item()
    return U, S


def reverse_affine_scan(a, b):
    """lam_t = a_t * lam_{t+1} + b_t, lam_T = 0 (Hillis-Steele, log T passes)."""
    a = a.flip(0).clone(); b = b.flip(0).clone()
    T = a.shape[0]; d = 1
    while d < T:
        ap = torch.ones_like(a); bp = torch.zeros_like(b)
        ap[d:] = a[:-d]; bp[d:] = b[:-d]
        b = a * bp + b; a = a * ap
        d *= 2
    return b.flip(0)


class LIFAffineFn(torch.autograd.Function):
    """Affine-reset LIF; grads w.r.t. I, u0, beta, gamma, delta (detached reset, sigmoid surrogate)."""

    @staticmethod
    def forward(ctx, I, u0, beta, gamma, delta, theta, alpha, mode, C, K):
        b, g, dl, th = float(beta), float(gamma), float(delta), float(theta)
        I = I.contiguous()
        if mode == "scan":
            U, S = aff_scan_forward(I, b, th, g, dl, u0, C=C, K=K)
        else:
            U, S = aff_serial_forward(I, b, th, g, dl, u0)
        ctx.save_for_backward(I, u0, U, S)
        ctx.cfg = (b, g, dl, th, alpha, mode)
        return S

    @staticmethod
    def backward(ctx, gS):
        I, u0, U, S = ctx.saved_tensors
        b, g, dl, th, alpha, mode = ctx.cfg
        uprev = torch.cat([u0[None], U[:-1]], 0)
        V = b * uprev + I
        sg = torch.sigmoid(alpha * (V - th)); sgd = alpha * sg * (1 - sg)
        J = (1 - S) + S * g                    # du_t/dv_t with detached spike
        a = torch.zeros_like(V); a[:-1] = b * J[:-1]
        lam = reverse_affine_scan(a, gS * sgd)  # lam_t = dL/dv_t
        lam_next = torch.cat([lam[1:], torch.zeros_like(lam[:1])], 0)
        dLdu = b * lam_next                     # dL/du_t
        return (lam, b * lam[0], (lam * uprev).sum().reshape(()), (S * V * dLdu).sum().reshape(()),
                (S * dLdu).sum().reshape(()), None, None, None, None, None)


def lif_affine(I, u0, beta, gamma, delta, theta=1.0, alpha=4.0, mode="scan", C=128, K=32):
    return LIFAffineFn.apply(I, u0, beta, gamma, delta, theta, alpha, mode, C, K)
