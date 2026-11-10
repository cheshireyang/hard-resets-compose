"""Differentiable hard-reset LIF with gradients w.r.t. the neuron parameters (beta, theta, u_r),
for fitting neuron models to long recordings.  Forward and backward use the serial or the scan
kernels (or auto dispatch); parameter gradients are read off the same adjoint lambda_t = dL/dv_t:
    dL/dbeta  = sum_t lambda_t u_{t-1}
    dL/dtheta = -sum_t gS_t sigma'(v_t - theta)               (detached reset)
    dL/du_r   = sum_t s_t * beta * lambda_{t+1}
"""
import torch
try:  # GPU kernels are optional (CPU prototype backend below)
    from . import triton_lif as L
except ImportError:  # pragma: no cover
    L = None


class LIFParamFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, I, u0, beta, theta, u_r, alpha, fwd_mode, bwd_mode, K):
        b, th, ur = float(beta), float(theta), float(u_r)
        I = I.contiguous()
        T = I.shape[0]
        if fwd_mode == "scan":
            U, S = L.scan_forward_v3(I, b, th, ur, u0, C=L._pick_C(T), K=K)
        else:
            U, S = L.serial_forward_tiled(I, b, th, ur, u0, BN=32, D=8, num_warps=1)
        ctx.save_for_backward(I, u0, U, S)
        ctx.cfg = (b, th, ur, alpha, bwd_mode)
        return S

    @staticmethod
    def backward(ctx, gS):
        I, u0, U, S = ctx.saved_tensors
        b, th, ur, alpha, bwd_mode = ctx.cfg
        gS = gS.contiguous(); GU = torch.zeros_like(gS)
        T = I.shape[0]
        if bwd_mode == "scan":
            lam = L.scan_backward_tiled(I, u0, U, S, gS, GU, b, th, ur, alpha, C=L._pick_C(T), BNr=32, D=8, nwr=1)
        else:
            lam = L.serial_backward_tiled(I, u0, U, S, gS, GU, b, th, ur, alpha, BN=32, D=4, num_warps=1)
        uprev = torch.cat([u0[None], U[:-1]], 0)
        v = b * uprev + I
        sg = L.surrogate_grad(v - th, alpha) if hasattr(L, "surrogate_grad") else None
        if sg is None:
            sgm = torch.sigmoid(alpha * (v - th)); sg = alpha * sgm * (1 - sgm)
        g_beta = (lam * uprev).sum()
        g_theta = -(gS * sg).sum()
        lam_next = torch.cat([lam[1:], torch.zeros_like(lam[:1])], 0)
        g_ur = (S * b * lam_next).sum()
        return lam, b * lam[0], g_beta.reshape(()), g_theta.reshape(()), g_ur.reshape(()), None, None, None, None


def lif_param(I, u0, beta, theta, u_r, alpha=4.0, mode="auto", K=16):
    T, N = I.shape
    if mode == "auto":
        fwd, bwd = L.auto_modes(N, T)
    else:
        fwd = bwd = mode
    return LIFParamFn.apply(I, u0, beta, theta, u_r, alpha, fwd, bwd, K)


# ----------------------------------------------------------------------------- CPU / prototype backend
from . import fast as _fast


class LIFParamFnCPU(torch.autograd.Function):
    """Same gradients as LIFParamFn, using the PyTorch prototype (any device, any dtype)."""

    @staticmethod
    def forward(ctx, I, u0, beta, theta, u_r, alpha, C, repair):
        b, th, ur = float(beta), float(theta), float(u_r)
        U, S, V = _fast.scan_forward(I.contiguous(), b, th, ur, u0, C=C, repair=repair)
        ctx.save_for_backward(I, u0, U, S, V)
        ctx.cfg = (b, th, ur, alpha)
        return S

    @staticmethod
    def backward(ctx, gS):
        I, u0, U, S, V = ctx.saved_tensors
        b, th, ur, alpha = ctx.cfg
        sg = _fast.surrogate_grad(V - th, alpha)
        J = 1 - S
        a = torch.zeros_like(V); a[:-1] = b * J[:-1]
        lam = _fast.reverse_affine_scan(a, gS * sg)
        uprev = torch.cat([u0[None], U[:-1]], 0)
        lam_next = torch.cat([lam[1:], torch.zeros_like(lam[:1])], 0)
        return (lam, b * lam[0], (lam * uprev).sum().reshape(()), -(gS * sg).sum().reshape(()),
                (S * b * lam_next).sum().reshape(()), None, None, None)


def lif_param_cpu(I, u0, beta, theta, u_r, alpha=4.0, C=64, repair=10 ** 6):
    return LIFParamFnCPU.apply(I, u0, beta, theta, u_r, alpha, C, repair)


class LIFSoftFn(torch.autograd.Function):
    """Soft (subtractive) reset LIF on the GPU, serial kernels; grads w.r.t. I, u0, beta."""

    @staticmethod
    def forward(ctx, I, u0, beta, theta, alpha):
        b, th = float(beta), float(theta)
        I = I.contiguous()
        U, S = L.soft_serial_forward(I, b, th, u0)
        ctx.save_for_backward(I, u0, U)
        ctx.cfg = (b, th, alpha)
        return S

    @staticmethod
    def backward(ctx, gS):
        I, u0, U = ctx.saved_tensors
        b, th, alpha = ctx.cfg
        lam = L.soft_serial_backward(I, u0, U, gS.contiguous(), b, th, alpha)
        uprev = torch.cat([u0[None], U[:-1]], 0)
        return lam, b * lam[0], (lam * uprev).sum().reshape(()), None, None


def lif_soft(I, u0, beta, theta=1.0, alpha=4.0):
    return LIFSoftFn.apply(I, u0, beta, theta, alpha)
