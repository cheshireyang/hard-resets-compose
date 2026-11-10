"""Is the serial baseline latency-bound?  Compare variants of the fused serial forward."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, triton, triton.language as tl
from lifscan import triton_lif as L

@triton.jit
def serial_tiled(I, U0, Uout, Sout, T, N, beta, theta, u_r, BN: tl.constexpr, D: tl.constexpr):
    pid = tl.program_id(0)
    n = pid * BN + tl.arange(0, BN)
    mask = n < N
    rows = tl.arange(0, D)
    u = tl.load(U0 + n, mask=mask, other=0.0)
    off = rows[:, None] * N + n[None, :]
    m2 = mask[None, :]
    xt = tl.load(I + off, mask=m2, other=0.0)
    for b in range(0, T // D):
        nb = b + 1
        xn = tl.load(I + nb * D * N + off, mask=m2 & (nb < T // D), other=0.0)  # prefetch next tile
        ut = tl.zeros([D, BN], dtype=tl.float32)
        st = tl.zeros([D, BN], dtype=tl.float32)
        for j in tl.static_range(D):
            x = tl.sum(tl.where(rows[:, None] == j, xt, 0.0), axis=0)
            u, s, v = L._step(u, x, beta, theta, u_r)
            ut = tl.where(rows[:, None] == j, u[None, :], ut)
            st = tl.where(rows[:, None] == j, s.to(tl.float32)[None, :], st)
        tl.store(Uout + b * D * N + off, ut, mask=m2)
        tl.store(Sout + b * D * N + off, st, mask=m2)
        xt = xn

def tm(fn, reps=10):
    fn(); torch.cuda.synchronize(); ts=[]
    for _ in range(reps):
        a=torch.cuda.Event(enable_timing=True); b=torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    return sorted(ts)[len(ts)//2]

for N, T in ((128, 16384), (1024, 16384), (8192, 16384), (65536, 4096)):
    I=(0.15+0.4*torch.randn(T,N,device="cuda")).contiguous(); u0=torch.zeros(N,device="cuda")
    Ur, Sr = L.serial_forward(I, 0.9, 1., 0., u0)
    line=f"N={N:<6} T={T:<6} base {tm(lambda: L.serial_forward(I,0.9,1.,0.,u0)):.3f}"
    for BN in (32, 128):
        for D in (4, 8, 16, 32):
            U=torch.empty_like(I); S=torch.empty_like(I)
            f=lambda: serial_tiled[(triton.cdiv(N,BN),)](I,u0,U,S,T,N,0.9,1.,0.,BN=BN,D=D)
            t=tm(f); eq=torch.equal(U,Ur) and torch.equal(S,Sr)
            line+=f" | BN{BN} D{D} {t:.3f}{'' if eq else '!'}"
    print(line, flush=True)
