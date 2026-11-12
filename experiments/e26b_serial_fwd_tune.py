"""E26b: how far was the serial FORWARD baseline from its best?  Hard, soft and ALIF serial forward kernels,
paper grid (E6c: four tiles; E17/E25: their grids) vs a full grid BN x D x num_warps.  Outputs identical by construction."""
import sys, os, json, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from lifscan import triton_lif as L, triton_soft as TS, triton_alif as A


def tm(fn, reps=5):
    fn(); torch.cuda.synchronize(); ts = []
    for _ in range(reps):
        a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    return sorted(ts)[len(ts) // 2]


PAPER = ((32, 4, 1), (64, 4, 2), (32, 8, 1), (128, 4, 4))
FULL = [(BN, D, nw) for BN in (16, 32, 64, 128) for D in (2, 4, 8, 16, 32) for nw in (1, 2, 4)]
rows = []
for N, T in itertools.product((4, 16, 128, 1024, 8192, 32768), (1024, 8192, 65536)):
    if N * T > 2 ** 28: continue
    torch.manual_seed(0)
    I = (0.15 + 0.4 * torch.randn(T, N, device="cuda")).contiguous(); u0 = torch.zeros(N, device="cuda")
    r = dict(N=N, T=T)
    ok = lambda BN, D: BN <= max(16, N) and T % D == 0
    hard = {cfg: tm(lambda: L.serial_forward_tiled(I, 0.9, 1.0, 0.0, u0, BN=cfg[0], D=cfg[1], num_warps=cfg[2])) for cfg in FULL if ok(*cfg[:2])}
    hard_plain = tm(lambda: L.serial_forward(I, 0.9, 1.0, 0.0, u0))
    r["hard_paper"] = min([hard_plain] + [tm(lambda: L.serial_forward_tiled(I, 0.9, 1.0, 0.0, u0, BN=c[0], D=c[1], num_warps=c[2])) for c in PAPER])
    kh = min(hard, key=hard.get); r["hard_full"] = min(hard[kh], hard_plain); r["hard_cfg"] = list(kh)
    soft = {cfg: tm(lambda: TS.soft_serial_forward_tiled(I, 0.9, 1.0, u0, BN=cfg[0], D=cfg[1], num_warps=cfg[2])) for cfg in FULL if ok(*cfg[:2])}
    ks = min(soft, key=soft.get); r["soft_full"] = soft[ks]; r["soft_cfg"] = list(ks)
    r["soft_paper"] = min([tm(lambda: TS.soft_serial_forward(I, 0.9, 1.0, u0))] +                     # E17 grid
                          [tm(lambda: TS.soft_serial_forward_tiled(I, 0.9, 1.0, u0, BN=c[0], D=c[1], num_warps=c[2]))
                           for c in ((32, 4, 1), (32, 8, 1), (64, 4, 2))])
    Ia = (0.3 * I).contiguous()
    al = {cfg: tm(lambda: A.alif_serial_forward(Ia, 0.95, 0.995, 0.9, 1.0, 0.0, u0, u0, BN=cfg[0], D=cfg[1], num_warps=cfg[2])) for cfg in FULL if ok(*cfg[:2])}
    ka = min(al, key=al.get); r["alif_full"] = al[ka]; r["alif_cfg"] = list(ka)
    r["alif_paper"] = min(tm(lambda: A.alif_serial_forward(Ia, 0.95, 0.995, 0.9, 1.0, 0.0, u0, u0, BN=BN, D=D))
                          for BN in (16, 32, 64) for D in (0, 4, 8) if BN <= max(16, N) and (D == 0 or T % D == 0))
    rows.append(r)
    print(f"N={N:6d} T={T:6d} | hard paper {r['hard_paper']:7.2f} full {r['hard_full']:7.2f} ({r['hard_paper'] / r['hard_full']:4.2f}x, {kh}) | "
          f"soft paper {r['soft_paper']:7.2f} full {r['soft_full']:7.2f} ({r['soft_paper'] / r['soft_full']:4.2f}x, {ks}) | "
          f"alif paper {r['alif_paper']:7.2f} full {r['alif_full']:7.2f} ({r['alif_paper'] / r['alif_full']:4.2f}x, {ka})", flush=True)
    json.dump(rows, open("results/e26b_serial_fwd_tune.json", "w"), indent=1)
