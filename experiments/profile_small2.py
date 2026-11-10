import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, triton
from lifscan import triton_lif as L
def tm(fn, reps=15):
    fn(); torch.cuda.synchronize(); ts=[]
    for _ in range(reps):
        a=torch.cuda.Event(enable_timing=True); b=torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    return sorted(ts)[len(ts)//2]
for beta,K in ((0.9,32),(0.99,64)):
  for N,T,C in ((128,65536,256),(1024,16384,128)):
    I=(0.15+0.4*torch.randn(T,N,device="cuda")).contiguous(); u0=torch.zeros(N,device="cuda")
    M=T//C; Imin=I.amin(0); inv=1/(1-beta)
    print(f"beta={beta} N={N} T={T} C={C} K={K} serial {tm(lambda: L.serial_forward_tiled(I,beta,1.,0.,u0,BN=32,D=8,num_warps=1)):.3f}")
    for BN,nw in ((128,4),(64,4),(32,4),(32,2),(32,1),(16,1),(16,2)):
        MA=torch.empty(M,N,device="cuda"); MB=torch.empty_like(MA); MX=torch.empty(M,K,N,device="cuda"); MC=torch.empty_like(MX)
        MN=torch.empty(M,N,device="cuda",dtype=torch.int32); OVF=torch.empty_like(MN); St=torch.empty(M,N,device="cuda")
        try:
            tb=tm(lambda: L.build2_kernel[(M,triton.cdiv(N,BN))](I,Imin,u0,MA,MB,MX,MC,MN,OVF,N,C,beta,1.,0.,inv,0,K=K,BN=BN,STAGES=1,num_warps=nw))
            tc=tm(lambda: L.carry2_kernel[(triton.cdiv(N,BN),)](I,u0,MA,MB,MX,MC,MN,OVF,St,N,M,C,beta,1.,0.,K=K,BN=BN,STAGES=1,num_warps=nw))
            print(f"   BN={BN:<4} nw={nw} build {tb:.3f} carry {tc:.3f} ovf {OVF.float().mean().item():.3f}", flush=True)
        except Exception as e: print("   err", BN, nw, repr(e)[:60])
