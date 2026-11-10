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
beta=0.9; dev="cuda"; N,T,C=65536,4096,64; M=T//C
I=(0.15+0.4*torch.randn(T,N,device=dev)).contiguous(); u0=torch.zeros(N,device=dev)
lo,hi=L.state_bounds(I,beta,1.,0.,u0)
for K in (16,32):
  for BN in (32,64,128,256,512):
    for nw in (1,2,4,8):
        MA=torch.empty(M,N,device=dev); MB=torch.empty_like(MA); MX=torch.empty(M,K,N,device=dev); MC=torch.empty_like(MX)
        MN=torch.empty(M,N,device=dev,dtype=torch.int32); OVF=torch.empty_like(MN)
        g2=(M,triton.cdiv(N,BN))
        try:
            t=tm(lambda: L.build_kernel[g2](I,lo,hi,MA,MB,MX,MC,MN,OVF,N,C,beta,1.,0.,K=K,BN=BN,num_warps=nw))
            print(f"K={K} BN={BN} nw={nw} build {t:.3f}", flush=True)
        except Exception as e: print("err",BN,nw,repr(e)[:80])
