import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from lifscan import triton_lif as L
def tm(fn, reps=20):
    fn(); torch.cuda.synchronize(); ts=[]
    for _ in range(reps):
        a=torch.cuda.Event(enable_timing=True); b=torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    return sorted(ts)[len(ts)//2]
beta=0.9
for N,T,C,K in ((128,65536,256,32),(1024,16384,256,32),(128,4096,64,16),(8192,16384,256,16)):
    I=(0.15+0.4*torch.randn(T,N,device="cuda")).contiguous(); u0=torch.zeros(N,device="cuda")
    Ur,Sr=L.serial_forward(I,beta,1.,0.,u0)
    t1=tm(lambda: L.scan_forward_tiled(I,beta,1.,0.,u0,C=C,K=K))
    line=f"N={N} T={T}: v1 {t1:.3f}"
    for st,cst in ((1,1),(2,1),(3,1),(3,2),(4,2)):
        U,S=L.scan_forward_v2(I,beta,1.,0.,u0,C=C,K=K,STAGES=st,CSTAGES=cst,repair=10**6)
        eq=torch.equal(U,Ur) and torch.equal(S,Sr)
        line+=f" | v2 {st},{cst} {tm(lambda: L.scan_forward_v2(I,beta,1.,0.,u0,C=C,K=K,STAGES=st,CSTAGES=cst)):.3f}{'' if eq else '!'}"
    print(line, flush=True)
