import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, triton
from lifscan import triton_lif as L
def tm(fn, reps=10):
    fn(); torch.cuda.synchronize(); ts=[]
    for _ in range(reps):
        a=torch.cuda.Event(enable_timing=True); b=torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    return sorted(ts)[len(ts)//2]
beta=0.9; dev="cuda"
for N, T in ((65536, 4096), (8192, 16384), (1024, 16384)):
    I=(0.15+0.4*torch.randn(T,N,device=dev)).contiguous(); u0=torch.zeros(N,device=dev)
    print(f"N={N} T={T} serial fwd {tm(lambda: L.serial_forward(I,beta,1.,0.,u0)):.3f}")
    for C,K,BN in ((64,16,128),(64,32,128),(64,32,64),(128,32,64),(256,32,64),(64,64,32)):
        M=T//C; lo,hi=L.state_bounds(I,beta,1.,0.,u0)
        MA=torch.empty(M,N,device=dev); MB=torch.empty_like(MA); MX=torch.empty(M,K,N,device=dev); MC=torch.empty_like(MX)
        MN=torch.empty(M,N,device=dev,dtype=torch.int32); OVF=torch.empty_like(MN)
        g2=(M,triton.cdiv(N,BN))
        tb=tm(lambda: L.build_kernel[g2](I,lo,hi,MA,MB,MX,MC,MN,OVF,N,C,beta,1.,0.,K=K,BN=BN))
        St=torch.empty(M,N,device=dev)
        tc=tm(lambda: L.carry_kernel[(triton.cdiv(N,BN),)](I,u0,MA,MB,MX,MC,MN,OVF,St,N,M,C,beta,1.,0.,K=K,BN=BN))
        U=torch.empty_like(I); S=torch.empty_like(I); E=torch.empty(M,N,device=dev)
        tr=tm(lambda: L.replay_kernel[g2](I,St,U,S,E,N,C,beta,1.,0.,BN=BN))
        print(f"  C={C:<4}K={K:<3}BN={BN:<4} build {tb:.3f} carry {tc:.3f} replay {tr:.3f} ovf {OVF.float().mean().item():.3f}")
