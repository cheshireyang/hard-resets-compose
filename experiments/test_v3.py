import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from lifscan import triton_lif as L
def tm(fn, reps=15):
    fn(); torch.cuda.synchronize(); ts=[]
    for _ in range(reps):
        a=torch.cuda.Event(enable_timing=True); b=torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    return sorted(ts)[len(ts)//2]
ok=True
for beta in (0.5, 0.9, 0.99, 1.0):
    for mu, sd in ((0.15, 0.4), (-0.1, 0.6), (0.4, 0.2)):
        T, N = 4096, 777
        I=(mu+sd*torch.randn(T,N,device="cuda")).contiguous(); u0=(0.3*torch.randn(N,device="cuda")).contiguous()
        Ur,Sr=L.serial_forward(I,beta,1.,-0.1,u0)
        for K in (4, 16):
            st={}
            U,S=L.scan_forward_v3(I,beta,1.,-0.1,u0,C=64,K=K,repair=10**6,stats=st)
            eq=torch.equal(U,Ur) and torch.equal(S,Sr); ok&=eq
            print(f"beta={beta} mu={mu} K={K} bit={eq} ovf={st['overflow_frac']:.3f} mean_pieces={st['mean_pieces']:.1f} max={st['max_pieces']} repair={st['repair_iters']}")
print("V3 OK" if ok else "FAIL")
for beta,K in ((0.9,8),(0.9,16),(0.99,16),(0.99,32)):
    for N,T,C in ((128,65536,256),(1024,16384,128),(8192,16384,128)):
        I=(0.15+0.4*torch.randn(T,N,device="cuda")).contiguous(); u0=torch.zeros(N,device="cuda")
        st={}; L.scan_forward_v3(I,beta,1.,0.,u0,C=C,K=K,stats=st)
        ts=tm(lambda: L.serial_forward_tiled(I,beta,1.,0.,u0,BN=32,D=8,num_warps=1))
        t3=tm(lambda: L.scan_forward_v3(I,beta,1.,0.,u0,C=C,K=K))
        print(f"beta={beta} K={K} N={N} T={T}: serial {ts:.3f} v3 {t3:.3f} x{ts/t3:.2f} ovf {st['overflow_frac']:.3f} pieces {st['mean_pieces']:.1f}", flush=True)
