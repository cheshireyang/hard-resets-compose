import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, triton
from lifscan import triton_lif as L
def tm(fn, reps=20):
    fn(); torch.cuda.synchronize(); ts=[]
    for _ in range(reps):
        a=torch.cuda.Event(enable_timing=True); b=torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    return sorted(ts)[len(ts)//2]
beta=0.9; dev="cuda"
for N,T,C,K in ((128,65536,256,32),(1024,16384,256,32),(128,4096,64,16)):
    I=(0.15+0.4*torch.randn(T,N,device=dev)).contiguous(); u0=torch.zeros(N,device=dev)
    M=T//C; BN=128
    tsb=tm(lambda: L.state_bounds(I,beta,1.,0.,u0)); lo,hi=L.state_bounds(I,beta,1.,0.,u0)
    MA=torch.empty(M,N,device=dev); MB=torch.empty_like(MA); MX=torch.empty(M,K,N,device=dev); MC=torch.empty_like(MX)
    MN=torch.empty(M,N,device=dev,dtype=torch.int32); OVF=torch.empty_like(MN)
    tb=tm(lambda: L.build_kernel[(M,triton.cdiv(N,BN))](I,lo,hi,MA,MB,MX,MC,MN,OVF,N,C,beta,1.,0.,K=K,BN=BN))
    St=torch.empty(M,N,device=dev)
    tc=tm(lambda: L.carry_kernel[(triton.cdiv(N,BN),)](I,u0,MA,MB,MX,MC,MN,OVF,St,N,M,C,beta,1.,0.,K=K,BN=BN))
    U=torch.empty_like(I); S=torch.empty_like(I); E=torch.empty(M,N,device=dev)
    tr=tm(lambda: L.fwd_tiled_kernel[(M,triton.cdiv(N,32))](I,St,U,S,E,N,C,beta,1.,0.,BN=32,D=4,num_warps=1))
    tot=tm(lambda: L.scan_forward_tiled(I,beta,1.,0.,u0,C=C,K=K))
    print(f"N={N} T={T} C={C} K={K}: bounds {tsb:.3f} build {tb:.3f} carry {tc:.3f} replay {tr:.3f} | total {tot:.3f}")
