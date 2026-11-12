"""E25b: ALIF backward (two affine scans) vs the fused serial backward, and forward+backward totals.
Same inputs as E25; the forward is the best E25 configuration of each cell (prev predictor), read from E25's JSON.
usage: python3 experiments/e25b_alif_fb.py E25.json OUT.json"""
import sys, os, math, json, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from lifscan import triton_alif as A


def tm(fn, reps=7):
    fn(); torch.cuda.synchronize(); ts = []
    for _ in range(reps):
        a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    return sorted(ts)[len(ts) // 2]


E25 = json.load(open(sys.argv[1])); OUT = sys.argv[2]
rows = []
for c in E25:
    beta, rho, q, mfree, N, T = c["beta"], c["rho"], c["q"], c["m"], c["N"], c["T"]
    torch.manual_seed(0)
    I = ((1 - beta) * mfree + 0.5 * math.sqrt(1 - beta * beta) * torch.randn(T, N, device="cuda")).contiguous()
    u0 = torch.zeros(N, device="cuda"); w0 = torch.zeros(N, device="cuda")
    U, W, S = A.alif_serial_forward(I, beta, rho, q, 1.0, 0.0, u0, w0, BN=32, D=8)
    GS = torch.randn(T, N, device="cuda")
    ser = {}
    for BN, D in itertools.product((16, 32, 64), (4, 8)):
        if BN > max(16, N): continue
        ser[(BN, D)] = tm(lambda: A.alif_backward(I, U, W, S, GS, beta, rho, q, 1.0, u0, w0, mode="serial", BN=BN, D=D))
    sc = {}
    for C, BN, D in itertools.product((32, 64, 128), (16, 32), (4, 8)):
        if BN > max(16, N): continue
        sc[(C, BN, D)] = tm(lambda: A.alif_backward(I, U, W, S, GS, beta, rho, q, 1.0, u0, w0, mode="scan", C=C, BN=BN, D=D))
    ks, kc = min(ser, key=ser.get), min(sc, key=sc.get)
    LVs, LWs = A.alif_backward(I, U, W, S, GS, beta, rho, q, 1.0, u0, w0, mode="serial", BN=ks[0], D=ks[1])
    LVc, LWc = A.alif_backward(I, U, W, S, GS, beta, rho, q, 1.0, u0, w0, mode="scan", C=kc[0], BN=kc[1], D=kc[2])
    relv = ((LVc - LVs).abs().max() / LVs.abs().max()).item(); relw = ((LWc - LWs).abs().max() / LWs.abs().max().clamp(min=1e-30)).item()
    r = dict(name=c["name"], beta=beta, rho=rho, q=q, m=mfree, N=N, T=T, rate=c["rate"],
             ms_bwd_serial=ser[ks], ms_bwd_scan=sc[kc], cfg_bwd_serial=list(ks), cfg_bwd_scan=list(kc), bwd_rel_err_v=relv, bwd_rel_err_w=relw,
             ms_fwd_serial=c["ms_serial"], ms_fwd_prev=c["ms_prev"], ms_fwd_warm4=c["ms_warm4"])
    r["fb_serial"] = r["ms_fwd_serial"] + r["ms_bwd_serial"]
    r["fb_prev"] = min(r["ms_fwd_prev"], r["ms_fwd_serial"]) + min(r["ms_bwd_scan"], r["ms_bwd_serial"])     # dispatching layer
    r["fb_scan_only"] = r["ms_fwd_prev"] + r["ms_bwd_scan"]
    rows.append(r)
    print(f"{r['name']:6s} m={mfree} N={N:5d} T={T}: bwd serial {r['ms_bwd_serial']:7.2f} scan {r['ms_bwd_scan']:7.2f} ({r['ms_bwd_serial'] / r['ms_bwd_scan']:5.2f}x, err {relv:.0e}) | "
          f"fwd+bwd serial {r['fb_serial']:7.2f} scan {r['fb_scan_only']:7.2f} ({r['fb_serial'] / r['fb_scan_only']:5.2f}x) dispatch {r['fb_serial'] / r['fb_prev']:5.2f}x", flush=True)
    json.dump(rows, open(OUT, "w"), indent=1)
