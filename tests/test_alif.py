"""GPU tests for the ALIF scan: tiled serial == plain serial (bit-exact); scan == serial after repair (bit-exact),
for oracle, previous-iterate and warm-up predictions of the entry adaptation, and with deliberately bad boxes."""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from lifscan import triton_alif as A

ok = True
dev = "cuda"
for name, beta, rho, q in (("lsnn", 0.95, 0.995, 0.9), ("glif2", 0.99, 0.96, 0.2), ("appD-a", 0.9, 0.95, 0.5)):
    for mfree in (0.5, 2.0):
        torch.manual_seed(0)
        T, N, C = 16384, 64, 128
        sig = 0.5
        I = ((1 - beta) * mfree + sig * math.sqrt(1 - beta * beta) * torch.randn(T, N, device=dev)).contiguous()
        u0 = torch.zeros(N, device=dev); w0 = (0.1 * torch.rand(N, device=dev)).contiguous()
        Ur, Wr, Sr = A.alif_serial_forward(I, beta, rho, q, 1.0, 0.0, u0, w0)
        Ut, Wt, St = A.alif_serial_forward(I, beta, rho, q, 1.0, 0.0, u0, w0, D=8)
        same_tiled = torch.equal(Ut, Ur) and torch.equal(Wt, Wr) and torch.equal(St, Sr)
        ok &= same_tiled
        Wentry = torch.cat([w0[None], Wr[C - 1:-1:C]], 0)            # true entry adaptation of every chunk
        _, Wprev, _ = A.alif_serial_forward((1.005 * I).contiguous(), beta, rho, q, 1.0, 0.0, u0, w0)
        Wpred_prev = torch.cat([w0[None], Wprev[C - 1:-1:C]], 0)
        Wpred_warm = A.alif_warm_predict(I, beta, rho, q, 1.0, 0.0, u0, w0, C, Wn=4)
        Wpred_bad = Wentry + 5 * q                                    # every box misses: the carry must go serial
        line = f"{name:6s} m={mfree}: rate {100 * Sr.mean().item():.2f}%  tiled==plain {same_tiled}"
        for tag, Wp in (("oracle", Wentry), ("prev", Wpred_prev), ("warm4", Wpred_warm), ("bad", Wpred_bad)):
            st = {}
            U, W, S = A.alif_scan_forward(I, beta, rho, q, 1.0, 0.0, u0, w0, Wp.contiguous(), 0.1 * q, C=C, K1=16, K2=32, stats=st)
            flips = (S != Sr).float().mean().item()
            st2 = {}
            U2, W2, S2 = A.alif_scan_forward(I, beta, rho, q, 1.0, 0.0, u0, w0, Wp.contiguous(), 0.1 * q, C=C, K1=16, K2=32,
                                              repair=10 ** 6, stats=st2)
            bit = torch.equal(U2, Ur) and torch.equal(W2, Wr) and torch.equal(S2, Sr)
            ok &= bit
            line += (f" | {tag}: miss {100 * st['miss_frac']:.1f}% ovf {100 * st['overflow_frac']:.1f}% pieces {st['mean_pieces']:.1f}"
                     f" lines {st['mean_lines']:.1f} flips {flips:.1e} repair {st2['repair_iters']} bit {bit}")
        print(line, flush=True)
print("ALIF OK" if ok else "ALIF FAIL")
