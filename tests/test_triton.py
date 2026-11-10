"""GPU tests (run on the CUDA box): python3 tests/test_triton.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from lifscan import triton_lif as tlif
from lifscan import fast

dev = "cuda"
ok = True
for beta in (0.5, 0.9, 0.99, 1.0):
    for mu, sd in ((0.15, 0.4), (-0.1, 0.6), (0.4, 0.2)):
        for K in (4, 16, 64):
            torch.manual_seed(0)
            T, N = 4096, 1000
            I = (mu + sd * torch.randn(T, N, device=dev)).contiguous()
            u0 = (0.3 * torch.randn(N, device=dev)).contiguous()
            Us, Ss = tlif.serial_forward(I, beta, 1.0, -0.1, u0)
            st = {}
            U, S = tlif.scan_forward(I, beta, 1.0, -0.1, u0, C=64, K=K, stats=st)
            st2 = {}
            U2, S2 = tlif.scan_forward(I, beta, 1.0, -0.1, u0, C=64, K=K, repair=10**6, stats=st2)
            bit = torch.equal(U2, Us) and torch.equal(S2, Ss)
            flip = (S != Ss).float().mean().item()
            ok &= bit
            print(f"beta={beta:<5} mu={mu:<5} K={K:<3} flip={flip:.2e} ovf={st['overflow_frac']:.3f} "
                  f"repair={st2['repair_iters']} bitexact={bit}")
# torch reference (float32 cpu-equivalent ops on gpu)
I = (0.15 + 0.4 * torch.randn(1024, 256, device=dev)); u0 = torch.zeros(256, device=dev)
ur, sr, _ = fast.serial_forward_nograd(I, 0.9, 1.0, 0.0, u0)
Us, Ss = tlif.serial_forward(I, 0.9, 1.0, 0.0, u0)
print("triton serial vs torch serial: spikes equal", torch.equal(sr, Ss), "max du", (ur - Us).abs().max().item())
# gradients
for detach in (True,):
    T, N = 2048, 512
    I = (0.15 + 0.4 * torch.randn(T, N, device=dev))
    u0 = torch.zeros(N, device=dev)
    U, S = tlif.serial_forward(I, 0.9, 1.0, 0.0, u0)
    GS = torch.randn_like(I); GU = torch.zeros_like(I)
    L1 = tlif.serial_backward(I, u0, U, S, GS, GU, 0.9, 1.0, 0.0)
    L2 = tlif.scan_backward(I, u0, U, S, GS, GU, 0.9, 1.0, 0.0, C=64)
    Id = I.double().requires_grad_(True)
    us, ss = fast.serial_lif(Id, 0.9, 1.0, 0.0, u0.double())
    (GS.double() * ss).sum().backward()
    ref = Id.grad
    print("bwd scan vs serial-triton rel", ((L1 - L2).norm() / L1.norm()).item(),
          "| serial-triton vs float64 autograd rel", ((L1.double() - ref).norm() / ref.norm()).item())
    ok &= ((L1 - L2).norm() / L1.norm()).item() < 1e-5
print("ALL OK" if ok else "FAIL")
