"""ctypes wrapper of csrc/lif_cpu.c (single-threaded CPU serial baseline for E31)."""
import ctypes, os, subprocess
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SO = os.path.join(_HERE, "csrc", "liblifcpu.so")


def _lib():
    src = os.path.join(_HERE, "csrc", "lif_cpu.c")
    if not os.path.exists(_SO) or os.path.getmtime(_SO) < os.path.getmtime(src):
        subprocess.check_call(["cc", "-O3", "-march=native", "-ffast-math", "-shared", "-fPIC", "-o", _SO, src, "-lm"])
    lib = ctypes.CDLL(_SO)
    f32p = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
    i64, f32 = ctypes.c_int64, ctypes.c_float
    lib.lif_fwd.argtypes = [f32p, f32p, f32p, f32p, i64, i64, f32, f32, f32]
    lib.lif_bwd.argtypes = [f32p, f32p, f32p, f32p, f32p, f32p, i64, i64, f32, f32, f32]
    return lib


LIB = None


def forward(I, u0, beta, theta=1.0, ur=0.0):
    global LIB
    LIB = LIB or _lib()
    T, N = I.shape
    U = np.empty_like(I); S = np.empty_like(I)
    LIB.lif_fwd(I, U, S, u0, T, N, beta, theta, ur)
    return U, S


def backward(I, U, S, GS, u0, beta, theta=1.0, alpha=4.0):
    global LIB
    LIB = LIB or _lib()
    T, N = I.shape
    Lam = np.empty_like(I)
    LIB.lif_bwd(I, U, S, GS, u0, Lam, T, N, beta, theta, alpha)
    return Lam
