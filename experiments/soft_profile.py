import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, triton
from lifscan import triton_soft as S, triton_lif as L
def tm(fn, reps=9):
    fn(); torch.cuda.synchronize(); ts=[]
    for _ in range(reps):
        a=torch.cuda.Event(enable_timing=True); b=torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    return sorted(ts)[len(ts)//2]
beta=0.95
for N,T in ((4,65536),(4,262144),(128,65536)):
    I=(0.15+0.4*torch.randn(T,N,device="cuda")).contiguous(); u0=torch.zeros(N,device="cuda")
    ts=min(tm(lambda: S.soft_serial_forward(I,beta,1.,u0)), tm(lambda: S.soft_serial_forward_tiled(I,beta,1.,u0)))
    for C,K in ((64,32),(128,32),(128,64),(256,64)):
        if T % C: continue
        M=T//C; BN=16
        EC=torch.empty(M,N,device="cuda"); EA=torch.empty_like(EC); EB=torch.empty_like(EC); Lo=torch.empty_like(EC); Hi=torch.empty_like(EC)
        te1=tm(lambda: L.env_chunk_kernel[(M,triton.cdiv(N,32))](I,EC,EA,EB,N,C,beta,0.0,BN=32,D=1,num_warps=1))
        te2=tm(lambda: L.env_carry_kernel[(triton.cdiv(N,16),)](u0,EC,EA,EB,Lo,N,M,1e-5,BN=16,num_warps=1))
        L.env_chunk_kernel[(M,triton.cdiv(N,32))](I,EC,EA,EB,N,C,beta,0.0,BN=32,D=1,num_warps=1); L.env_carry_kernel[(triton.cdiv(N,16),)](u0,EC,EA,EB,Lo,N,M,1e-5,BN=16,num_warps=1)
        S.soft_env_hi_chunk_kernel[(M,triton.cdiv(N,32))](I,EC,EA,EB,N,C,beta,1.0,BN=32,num_warps=1); S.soft_env_hi_carry_kernel[(triton.cdiv(N,16),)](u0,EC,EA,EB,Hi,N,M,1e-5,BN=16,num_warps=1)
        MA=torch.empty(M,N,device="cuda"); MXL=torch.empty(M,K,N,device="cuda"); MXR=torch.empty_like(MXL); MC=torch.empty_like(MXL)
        MN=torch.empty(M,N,device="cuda",dtype=torch.int32); OVF=torch.empty_like(MN)
        tb=tm(lambda: S.soft_build_kernel[(M,triton.cdiv(N,BN))](I,Lo,Hi,MA,MXL,MXR,MC,MN,OVF,N,C,beta,1.0,K=K,BN=BN,LOGK=int(math.log2(K)),num_warps=1))
        St=torch.empty(M,N,device="cuda")
        tc=tm(lambda: S.soft_carry_kernel[(triton.cdiv(N,16),)](I,u0,MA,MXL,MXR,MC,MN,OVF,St,N,M,C,beta,1.0,K=K,BN=16,num_warps=1))
        tot=tm(lambda: S.soft_scan_forward(I,beta,1.,u0,C=C,K=K))
        print(f"N={N} T={T} C={C} K={K}: serial {ts:.3f} | env-chunk {te1:.3f} env-carry {te2:.3f} build {tb:.3f} carry {tc:.3f} ovf {OVF.float().mean().item():.3f} | total {tot:.3f} x{ts/tot:.2f}", flush=True)
