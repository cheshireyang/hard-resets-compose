"""Run: python3 tests/test_fast.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from lifscan.fast import *

def gen(T, N, dt, seed, mu=0.15, sd=0.4):
    g = torch.Generator().manual_seed(seed)
    return (mu + sd * torch.randn(T, N, generator=g)).to(dt)

def check_forward():
    for dt in (torch.float64, torch.float32):
        for beta in (0.5, 0.9, 0.99, 1.0):
            for u_r in (0.0, -0.2):
                for C in (8, 64):
                    I = gen(1024, 64, dt, 0); u0 = torch.zeros(64, dtype=dt)
                    us, ss, vs = serial_forward_nograd(I, beta, 1.0, u_r, u0)
                    st = {}
                    u, s, v = scan_forward(I, beta, 1.0, u_r, u0, C=C, stats=st)
                    agree = (s == ss).double().mean().item()
                    st2 = {}
                    u2, s2, v2 = scan_forward(I, beta, 1.0, u_r, u0, C=C, repair=10**6, stats=st2)
                    bit = torch.equal(u2, us) and torch.equal(s2, ss)
                    print(f"{str(dt)[6:]:8s} beta={beta:<5} u_r={u_r:<5} C={C:<3} spike-agree={agree:.6f} "
                          f"maxdu={(u-us).abs().max().item():.2e} repair_iters={st2['repair_iters']} bitexact_after_repair={bit}")
                    assert bit

def check_grad():
    torch.manual_seed(0)
    for detach in (True, False):
        I = gen(512, 16, torch.float64, 1).requires_grad_(True)
        u0 = torch.zeros(16, dtype=torch.float64, requires_grad=True)
        w1 = torch.randn(512, 16, dtype=torch.float64); w2 = torch.randn(512, 16, dtype=torch.float64)
        us, ss = serial_lif(I, 0.9, 1.0, 0.0, u0, detach_reset=detach)
        L = (w1 * ss).sum() + (w2 * us).sum()
        gI, g0 = torch.autograd.grad(L, (I, u0))
        u, s = scan_lif(I, 0.9, 1.0, 0.0, u0, C=32, detach_reset=detach, repair=10**6)
        L2 = (w1 * s).sum() + (w2 * u).sum()
        hI, h0 = torch.autograd.grad(L2, (I, u0))
        rel = ((gI - hI).norm() / gI.norm()).item()
        print(f"grad detach_reset={detach}: rel err dI={rel:.2e}  du0={((g0-h0).norm()/g0.norm()).item():.2e}")
        assert rel < 1e-10

if __name__ == "__main__" and "--pruned" not in sys.argv:
    check_forward(); check_grad(); print("ALL OK")

def check_pruned():
    for dt in (torch.float64, torch.float32):
        for beta in (0.5, 0.9, 0.99, 1.0):
            for (mu, sd) in ((0.15, 0.4), (-0.1, 0.6), (0.4, 0.2)):
                for K in (2, 16):
                    I = gen(2048, 64, dt, 3, mu, sd); u0 = (0.3 * torch.randn(64)).to(dt)
                    us, ss, vs = serial_forward_nograd(I, beta, 1.0, -0.1, u0)
                    st = {}
                    u, s, v = scan_forward_pruned(I, beta, 1.0, -0.1, u0, C=64, K=K, stats=st)
                    agree = (s == ss).double().mean().item()
                    u2, s2, _ = scan_forward_pruned(I, beta, 1.0, -0.1, u0, C=64, K=K, repair=10**6, stats=st)
                    bit = torch.equal(u2, us) and torch.equal(s2, ss)
                    print(f"pruned {str(dt)[6:]:8s} beta={beta:<5} mu={mu:<5} K={K:<3} agree={agree:.6f} "
                          f"overflow={st['overflow_frac']:.3f} repair={st['repair_iters']} bit={bit}")
                    assert bit and agree > 0.999

if __name__ == "__main__":
    check_pruned(); print("PRUNED OK")
