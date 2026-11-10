"""Vectorised float implementation (PyTorch) of the map-composition scan.

Layout: inputs ``I`` of shape [T, N] (N = flattened batch x neurons).
Scalars beta, theta, u_r are python floats (shared); u0 is [N].

Forward (two-level scan, exact in exact arithmetic):
  1. split T into M = T/C chunks; build each chunk's segment map by a
     vectorised left fold over its C steps (all chunks & neurons at once);
  2. carry: evaluate chunk maps on scalar states, M sequential steps
     (or ``carry="tree"``: log-depth Hillis-Steele over the chunk maps);
  3. replay: C serial steps run in parallel for all chunks from the carried
     start states -> u_t, s_t, v_t for every t.
  Optional ``repair``: re-run replay with chunk starts replaced by the replayed
  end states of the previous chunk until nothing changes; the result is then
  bit-identical to the serial float loop (see docs/theory.md, Prop. F1).

Backward: the surrogate-gradient adjoint is the linear recurrence
  lam_t = a_t * lam_{t+1} + b_t,
solved with a log-depth Hillis-Steele affine scan.
"""
from __future__ import annotations

import math
from typing import Optional

import torch

# ----------------------------------------------------------------------------
# reference serial implementation (autograd through a python loop)
# ----------------------------------------------------------------------------


class SpikeFn(torch.autograd.Function):
    """Heaviside forward, surrogate derivative backward."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g * surrogate_grad(x, ctx.alpha), None


def surrogate_grad(x, alpha):
    # derivative of sigmoid(alpha * x)
    sg = torch.sigmoid(alpha * x)
    return alpha * sg * (1 - sg)


def serial_lif(I, beta, theta, u_r, u0, alpha=4.0, detach_reset=True):
    """Plain serial hard-reset LIF; differentiable through autograd."""
    T = I.shape[0]
    u = u0
    us, ss = [], []
    for t in range(T):
        v = beta * u + I[t]
        s = SpikeFn.apply(v - theta, alpha)
        sr = s.detach() if detach_reset else s
        u = v * (1 - sr) + u_r * sr
        us.append(u)
        ss.append(s)
    return torch.stack(us), torch.stack(ss)


@torch.no_grad()
def serial_forward_nograd(I, beta, theta, u_r, u0):
    T = I.shape[0]
    u = u0.clone()
    us = torch.empty_like(I)
    ss = torch.empty_like(I)
    vs = torch.empty_like(I)
    for t in range(T):
        v = beta * u + I[t]
        s = v >= theta
        u = torch.where(s, torch.full_like(v, u_r), v)
        us[t], ss[t], vs[t] = u, s.to(I.dtype), v
    return us, ss, vs


# ----------------------------------------------------------------------------
# chunk maps
# ----------------------------------------------------------------------------
# A map is stored as  (A [..], B [..], xd [.., K], cd [.., K], n [..])
#   xd = breakpoints in DESCENDING order (xd[0] largest), valid for idx < n
#   cd[i] = constant on [xd[i], xd[i-1])   (xd[-1] := +inf)
#   M(u) = A u + B  if n == 0 or u < xd[n-1]
#          cd[k]    where k = #{i < n : xd[i] > u}


def _safe_div(num, den):
    """num/den for den >= 0 that may underflow to 0: sign(num)*inf."""
    q = num / den
    inf = torch.full_like(q, math.inf)
    q = torch.where(den == 0, torch.where(num > 0, inf, -inf), q)
    return q


@torch.no_grad()
def build_chunk_maps(Ic, beta, theta, u_r):
    """Ic: [C, M, N]  -> maps with leading dims [M, N], capacity C."""
    C, M, N = Ic.shape
    dt, dev = Ic.dtype, Ic.device
    A = torch.ones(M, N, dtype=dt, device=dev)
    B = torch.zeros(M, N, dtype=dt, device=dev)
    xd = torch.full((M, N, C), math.inf, dtype=dt, device=dev)
    cd = torch.zeros(M, N, C, dtype=dt, device=dev)
    n = torch.zeros(M, N, dtype=torch.long, device=dev)
    xmin = torch.full((M, N), math.inf, dtype=dt, device=dev)
    ar = torch.arange(C, device=dev)
    for t in range(C):
        It = Ic[t]
        # existing constants pushed through the step (same arithmetic as serial)
        v = beta * cd + It[..., None]
        cd = torch.where(v >= theta, torch.full_like(v, u_r), v)
        # new breakpoint: preimage of the step's threshold under the affine part
        # u_prev = A u + B >= c_t  <=>  beta*(A u + B) + I_t >= theta
        # solved as u >= (theta - I_t - beta*B) / (beta*A)
        z = _safe_div(theta - It - beta * B, beta * A)
        new = z < xmin
        slot = (ar[None, None, :] == n[..., None]) & new[..., None]
        xd = torch.where(slot, z[..., None], xd)
        cd = torch.where(slot, torch.full_like(cd, u_r), cd)
        n = n + new.long()
        xmin = torch.where(new, z, xmin)
        A = beta * A
        B = beta * B + It
    return A, B, xd, cd, n


@torch.no_grad()
def state_bounds(I, beta, theta, u_r, u0):
    """Per-neuron [lo, hi) containing every state u_t, t>=0 (cheap, loose)."""
    Imin = I.min(0).values.clamp(max=0)
    base = torch.minimum(u0, torch.full_like(u0, u_r)).clamp(max=0)
    if beta < 1:
        lo = base + Imin / (1 - beta)
    else:
        lo = base + I.clamp(max=0).sum(0)
    hi = torch.maximum(u0, torch.full_like(u0, max(theta, u_r)))
    return lo, hi


@torch.no_grad()
def build_chunk_maps_pruned(Ic, beta, theta, u_r, lo, hi, K: int = 16):
    """Like build_chunk_maps but keeps only breakpoints relevant on the state
    domain [lo, hi]:
      * a new breakpoint z > hi is never reachable -> skipped;
      * once some breakpoint z <= lo exists, the map is constant-piecewise on
        the whole domain below the next breakpoint; a newer, smaller z' only
        changes the map below lo, i.e. outside the domain -> skipped.
    Capacity K; ``overflow`` marks (chunk, neuron) pairs that needed more."""
    C, M, N = Ic.shape
    dt, dev = Ic.dtype, Ic.device
    lo = lo[None].expand(M, N)
    hi = hi[None].expand(M, N)
    A = torch.ones(M, N, dtype=dt, device=dev)
    B = torch.zeros(M, N, dtype=dt, device=dev)
    xd = torch.full((M, N, K), math.inf, dtype=dt, device=dev)
    cd = torch.zeros(M, N, K, dtype=dt, device=dev)
    n = torch.zeros(M, N, dtype=torch.long, device=dev)
    xmin = torch.full((M, N), math.inf, dtype=dt, device=dev)
    overflow = torch.zeros(M, N, dtype=torch.bool, device=dev)
    ar = torch.arange(K, device=dev)
    for t in range(C):
        It = Ic[t]
        v = beta * cd + It[..., None]
        cd = torch.where(v >= theta, torch.full_like(v, u_r), v)
        z = _safe_div(theta - It - beta * B, beta * A)
        new = (z < xmin) & (z <= hi) & (xmin > lo)
        overflow |= new & (n >= K)
        new = new & (n < K)
        slot = (ar[None, None, :] == n[..., None]) & new[..., None]
        xd = torch.where(slot, z[..., None], xd)
        cd = torch.where(slot, torch.full_like(cd, u_r), cd)
        n = n + new.long()
        xmin = torch.where(new, z, xmin)
        A = beta * A
        B = beta * B + It
    return A, B, xd, cd, n, overflow


@torch.no_grad()
def scan_forward_pruned(I, beta, theta, u_r, u0, C=64, K=16, repair=0, stats=None):
    """Two-level scan with domain-pruned, capacity-K chunk maps.  Chunks whose
    map overflowed K get their start state from the replay of the previous
    chunk (serial fallback through the repair loop), so the result is still
    exact; ``stats['overflow_frac']`` reports how often that happens."""
    T, N = I.shape
    M = T // C
    Ic = I.reshape(M, C, N).transpose(0, 1).contiguous()
    lo, hi = state_bounds(I, beta, theta, u_r, u0)
    A, B, xd, cd, n, ovf = build_chunk_maps_pruned(Ic, beta, theta, u_r, lo, hi, K)
    starts = torch.empty(M, N, dtype=I.dtype, device=I.device)
    u = u0
    ovf_any = ovf.any(1).tolist()
    for m in range(M):
        starts[m] = u
        nu = eval_map(A[m], B[m], xd[m], cd[m], n[m], u)
        if ovf_any[m]:  # rare: serial fallback through this chunk
            w = u
            for t in range(C):
                vv = beta * w + Ic[t, m]
                w = torch.where(vv >= theta, torch.full_like(vv, u_r), vv)
            nu = torch.where(ovf[m], w, nu)
        u = nu
    us, ss, vs = replay(Ic, beta, theta, u_r, starts)
    it = 0
    for it in range(1, repair + 1):
        new_starts = torch.cat([u0[None], us[-1][:-1]], 0)
        if torch.equal(new_starts, starts):
            it -= 1
            break
        starts = new_starts
        us, ss, vs = replay(Ic, beta, theta, u_r, starts)
    if stats is not None:
        stats["repair_iters"] = it
        stats["overflow_frac"] = float(ovf.float().mean().item())
        stats["max_pieces"] = int(n.max().item()) + 1
    f = lambda z: z.transpose(0, 1).reshape(T, N)
    return f(us), f(ss), f(vs)


@torch.no_grad()
def serial_backward(v, s, gs, gu, beta, theta, u_r, alpha=4.0, detach_reset=True):
    """Hand-written serial adjoint (fair baseline for the scan backward)."""
    sg = surrogate_grad(v - theta, alpha)
    J = 1 - s
    if not detach_reset:
        J = J + (u_r - v) * sg
    lam = torch.zeros_like(v)
    nxt = torch.zeros_like(v[0])
    for t in range(v.shape[0] - 1, -1, -1):
        nxt = gs[t] * sg[t] + J[t] * (gu[t] + beta * nxt)
        lam[t] = nxt
    return lam


@torch.no_grad()
def eval_map(A, B, xd, cd, n, u):
    """Evaluate maps (leading dims [..]) at states u [..]."""
    K = xd.shape[-1]
    valid = torch.arange(K, device=u.device) < n[..., None]
    k = ((xd > u[..., None]) & valid).sum(-1)
    aff = k == n
    kc = k.clamp(max=K - 1)
    c = torch.gather(cd, -1, kc[..., None])[..., 0]
    return torch.where(aff, A * u + B, c)


@torch.no_grad()
def replay(Ic, beta, theta, u_r, starts):
    """Ic [C, M, N], starts [M, N] -> u, s, v each [C, M, N]."""
    C = Ic.shape[0]
    u = starts
    us = torch.empty_like(Ic)
    ss = torch.empty_like(Ic)
    vs = torch.empty_like(Ic)
    for t in range(C):
        v = beta * u + Ic[t]
        s = v >= theta
        u = torch.where(s, torch.full_like(v, u_r), v)
        us[t], ss[t], vs[t] = u, s.to(Ic.dtype), v
    return us, ss, vs


@torch.no_grad()
def scan_forward(I, beta, theta, u_r, u0, C: int = 64, repair: int = 0,
                 stats: Optional[dict] = None):
    """I [T, N] (T divisible by C), u0 [N] -> u, s, v each [T, N]."""
    T, N = I.shape
    assert T % C == 0, "pad T to a multiple of C"
    M = T // C
    Ic = I.reshape(M, C, N).transpose(0, 1).contiguous()  # [C, M, N]
    A, B, xd, cd, n = build_chunk_maps(Ic, beta, theta, u_r)
    starts = torch.empty(M, N, dtype=I.dtype, device=I.device)
    u = u0
    for m in range(M):
        starts[m] = u
        u = eval_map(A[m], B[m], xd[m], cd[m], n[m], u)
    us, ss, vs = replay(Ic, beta, theta, u_r, starts)
    it = 0
    for it in range(1, repair + 1):
        ends = us[-1]  # [M, N] replayed end state of each chunk
        new_starts = torch.cat([u0[None], ends[:-1]], 0)
        if torch.equal(new_starts, starts):
            it -= 1
            break
        starts = new_starts
        us, ss, vs = replay(Ic, beta, theta, u_r, starts)
    if stats is not None:
        stats["repair_iters"] = it
        stats["max_pieces"] = int(n.max().item()) + 1
        stats["mean_pieces"] = float(n.float().mean().item()) + 1
    f = lambda z: z.transpose(0, 1).reshape(T, N)
    return f(us), f(ss), f(vs)


# ----------------------------------------------------------------------------
# backward: linear time-varying recurrence via Hillis-Steele affine scan
# ----------------------------------------------------------------------------


def reverse_affine_scan(a, b):
    """Solve lam_t = a_t lam_{t+1} + b_t (t = T-1..0), lam_T = 0. a,b [T, N]."""
    # flip to forward form y_t = a_t y_{t-1} + b_t
    a = a.flip(0)
    b = b.flip(0)
    T = a.shape[0]
    d = 1
    while d < T:
        a_prev = torch.ones_like(a)
        b_prev = torch.zeros_like(b)
        a_prev[d:] = a[:-d]
        b_prev[d:] = b[:-d]
        # compose (a_prev,b_prev) then (a,b):  y = a*(a_prev*y + b_prev) + b
        b = a * b_prev + b
        a = a * a_prev
        d *= 2
    return b.flip(0)


class ScanLIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, I, u0, beta, theta, u_r, alpha, C, detach_reset, repair, K):
        if K is None:
            u, s, v = scan_forward(I, beta, theta, u_r, u0, C=C, repair=repair)
        else:
            u, s, v = scan_forward_pruned(I, beta, theta, u_r, u0, C=C, K=K, repair=repair)
        ctx.save_for_backward(v, s)
        ctx.cfg = (beta, theta, u_r, alpha, detach_reset)
        return u, s

    @staticmethod
    def backward(ctx, gu, gs):
        v, s = ctx.saved_tensors
        beta, theta, u_r, alpha, detach_reset = ctx.cfg
        sg = surrogate_grad(v - theta, alpha)
        J = 1 - s
        if not detach_reset:
            J = J + (u_r - v) * sg
        if gu is None:
            gu = torch.zeros_like(v)
        if gs is None:
            gs = torch.zeros_like(v)
        # lam_t = dL/dv_t = gs_t sg_t + J_t (gu_t + beta lam_{t+1})
        a = torch.zeros_like(v)
        a[:-1] = beta * J[:-1]  # coefficient multiplying lam_{t+1}
        b = gs * sg + J * gu
        lam = reverse_affine_scan(a, b)
        gI = lam
        gu0 = beta * lam[0]
        return gI, gu0, None, None, None, None, None, None, None, None


def scan_lif(I, beta, theta, u_r, u0, alpha=4.0, C=64, detach_reset=True, repair=0, K=None):
    return ScanLIF.apply(I, u0, beta, theta, u_r, alpha, C, detach_reset, repair, K)
