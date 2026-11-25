"""E27c: Exact 1-to-1 spike-matching F1 vs many-to-one window F1 on Allen Cell Types recordings.

Checks whether the spike-timing ranking of Hard LIF, Soft LIF, ALIF, and official Allen GLIF 1-5
changes when replacing the max_pool1d window F1 (which allows multiple predicted spikes to match one
recorded spike) with exact optimal 1-to-1 chronological bipartite matching within +-4 ms and +-10 ms.

For two sorted spike-index arrays a_1 < ... < a_m and b_1 < ... < b_n and tolerance w >= 0,
the greedy two-pointer algorithm (always match the earliest eligible pair) is provably optimal
(maximum-cardinality matching in a convex bipartite / unit-interval graph).

Usage:
  python3 experiments/e27c_1to1_f1.py [--data data/allen] [--spk results/e27all_spikes] [--out results/e27c_1to1_f1.json]
"""
import argparse, glob, json, math, os, sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifscan import triton_alif as A
from lifscan import triton_lif as L
from lifscan import triton_soft as TS

DT_MS = 0.2
dev = "cuda"


def load_noise2(path):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    sweeps = sorted({k.split("__")[0] for k in z.files if "__" in k})
    pairs = []
    for s in sweeps:
        if str(z[f"{s}__name"]) != "Noise 2":
            continue
        rate = float(z[f"{s}__rate"])
        i_pa = z[f"{s}__i_pa"]
        sp = z[f"{s}__spikes"]
        step = int(round(rate * DT_MS / 1000.0))
        n = len(i_pa) // step
        x = i_pa[: n * step].reshape(n, step).mean(1)
        s_bin = np.zeros(n, dtype=np.float32)
        idx = np.minimum((sp * 1000.0 / DT_MS).astype(np.int64), n - 1)
        s_bin[idx] = 1.0
        pairs.append((x, s_bin))
    if len(pairs) < 2:
        return None
    T = (min(len(p[0]) for p in pairs) // 256) * 256
    X = np.stack([p[0][:T] for p in pairs], 1).astype(np.float32) / 100.0
    S = np.stack([p[1][:T] for p in pairs], 1).astype(np.float32)
    return meta, torch.tensor(X, device=dev), torch.tensor(S, device=dev)


def f1_window(Sm, St, win):
    """Original many-to-one window F1 from e15_allen_fit.py."""
    pool = lambda M: torch.nn.functional.max_pool1d(M.T[:, None, :], 2 * win + 1, stride=1, padding=win)[:, 0, :].T
    rec = (pool(Sm) * St).sum() / St.sum().clamp(min=1)
    prec = (pool(St) * Sm).sum() / Sm.sum().clamp(min=1)
    return float(2 * rec * prec / (rec + prec + 1e-12))


def match_1to1_counts(idx_m, idx_t, win):
    """Provably maximum-cardinality 1-to-1 matching between sorted int arrays within +-win."""
    i = j = matched = 0
    nm, nt = len(idx_m), len(idx_t)
    while i < nm and j < nt:
        d = idx_m[i] - idx_t[j]
        if abs(d) <= win:
            matched += 1
            i += 1
            j += 1
        elif d < -win:
            i += 1
        else:
            j += 1
    return matched


def f1_1to1(Sm, St, win):
    """Exact 1-to-1 matching F1 pooled across the test repeats (same pooling as f1_window)."""
    Sm_np = Sm.detach().cpu().numpy()
    St_np = St.detach().cpu().numpy()
    matched = nm_tot = nt_tot = 0
    for col in range(St_np.shape[1]):
        im = np.flatnonzero(Sm_np[:, col] > 0.5)
        it = np.flatnonzero(St_np[:, col] > 0.5)
        matched += match_1to1_counts(im, it, win)
        nm_tot += len(im)
        nt_tot += len(it)
    if nm_tot == 0 or nt_tot == 0 or matched == 0:
        return 0.0
    rec = matched / nt_tot
    prec = matched / nm_tot
    return float(2.0 * rec * prec / (rec + prec))


def trains_from_times(times, T, n_cols):
    S = torch.zeros(T, n_cols, device=dev)
    for j, sp in enumerate(times[:n_cols]):
        idx = (np.asarray(sp) * 1000.0 / DT_MS).astype(np.int64)
        S[idx[(idx >= 0) & (idx < T)], j] = 1.0
    return S


def boot_ci(diffs, n_boot=10000, seed=0):
    arr = np.asarray(diffs, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    meds = np.median(arr[idx], axis=1)
    return dict(
        n=int(len(arr)),
        median=float(np.median(arr)),
        ci_lo=float(np.quantile(meds, 0.025)),
        ci_hi=float(np.quantile(meds, 0.975)),
        wins=int((arr > 0).sum()),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/allen")
    ap.add_argument("--spk", default="results/e27all_spikes")
    ap.add_argument("--out", default="results/e27c_1to1_f1.json")
    args = ap.parse_args()

    base = {}
    for pat in ("results/e15all_allen_w*.json",):
        for f in sorted(glob.glob(pat)):
            for r in json.load(open(f)):
                base[r["cell"]["specimen"]] = r
    alif = {}
    for f in sorted(glob.glob("results/e28all_allen_alif_w*.json")):
        for r in json.load(open(f)):
            if not r.get("failed"):
                alif[r["cell"]["specimen"]] = r
    off = {r["specimen"]: r for r in json.load(open("results/e27all_glif_official.json"))}

    w4, w10 = int(4.0 / DT_MS), int(10.0 / DT_MS)
    rows = []
    for path in sorted(glob.glob(f"{args.data}/*.npz")):
        sid = int(os.path.basename(path)[:-4])
        if sid not in base or sid not in alif or sid not in off:
            continue
        loaded = load_noise2(path)
        if loaded is None:
            continue
        meta, Xte, Ste = loaded
        if float(Ste.sum().item()) == 0.0:
            continue
        T, N = Xte.shape
        u0 = torch.zeros(N, device=dev)
        preds = {}
        b = base[sid]["models"]
        with torch.no_grad():
            mh = b["hard"]
            if math.isfinite(mh["beta"]) and 0 < mh["beta"] < 1:
                Ih = (mh["gain"] * Xte + mh["bias"]).contiguous()
                preds["hard"] = L.serial_forward_auto(Ih, mh["beta"], 1.0, mh["ur"], u0)[1]
            ms = b["soft"]
            if math.isfinite(ms["beta"]) and 0 < ms["beta"] < 1:
                Is = (ms["gain"] * Xte + ms["bias"]).contiguous()
                BN, D, _ = L.serial_fwd_cfg(N)
                while T % D: D //= 2
                preds["soft"] = TS.soft_serial_forward_tiled(Is, ms["beta"], 1.0, u0, BN=BN, D=D, num_warps=1)[1]
            a = alif[sid]
            Ia = (a["gain"] * Xte + a["bias"]).contiguous()
            preds["alif"] = A.alif(Ia, u0, u0, a["beta"], a["rho"], a["q"], 1.0, a["ur"], fwd="serial")[0]

        for lvl in off[sid]["levels"]:
            sp_file = f"{args.spk}/{sid}_glif{lvl}.npz"
            if not os.path.exists(sp_file):
                continue
            z = np.load(sp_file)
            preds[f"glif{lvl}"] = trains_from_times([z[k] for k in z.files], T, N)

        metrics = {}
        for k, Sm in preds.items():
            metrics[k] = dict(
                f1_4_win=f1_window(Sm, Ste, w4),
                f1_4_1to1=f1_1to1(Sm, Ste, w4),
                f1_10_win=f1_window(Sm, Ste, w10),
                f1_10_1to1=f1_1to1(Sm, Ste, w10),
                rate_ratio=float(Sm.sum().item() / max(Ste.sum().item(), 1.0)),
            )
        rows.append(dict(specimen=sid, dendrite=meta.get("dendrite", ""), metrics=metrics))
        if "glif5" in metrics:
            print(
                f"cell {sid} ({meta.get('dendrite','')[:6]:6s}) | ALIF vs GLIF5 F1_4: "
                f"win {metrics['alif']['f1_4_win']:.3f} vs {metrics['glif5']['f1_4_win']:.3f} | "
                f"1to1 {metrics['alif']['f1_4_1to1']:.3f} vs {metrics['glif5']['f1_4_1to1']:.3f}",
                flush=True,
            )

    # Compute summary stats and bootstrap 95% CIs
    g13 = [r for r in rows if all(k in r["metrics"] for k in ("hard", "soft", "alif", "glif1", "glif3"))]
    g5 = [r for r in g13 if all(k in r["metrics"] for k in ("glif2", "glif4", "glif5"))]
    summary = dict(n_all=len(g13), n_g5=len(g5), medians_all={}, medians_g5={}, paired={})
    for m in ("f1_4_win", "f1_4_1to1", "f1_10_win", "f1_10_1to1"):
        for k in ("hard", "soft", "alif", "glif1", "glif3"):
            summary["medians_all"].setdefault(k, {})[m] = float(np.median([r["metrics"][k][m] for r in g13])) if g13 else None
        for k in ("hard", "soft", "alif", "glif1", "glif2", "glif3", "glif4", "glif5"):
            summary["medians_g5"].setdefault(k, {})[m] = float(np.median([r["metrics"][k][m] for r in g5])) if g5 else None
    for tag, subset, a_, b_ in (
        ("alif_vs_glif5", g5, "alif", "glif5"),
        ("alif_vs_glif1", g13, "alif", "glif1"),
        ("alif_vs_hard", g13, "alif", "hard"),
        ("soft_vs_hard", g13, "soft", "hard"),
    ):
        if subset:
            summary["paired"][tag] = {
                m: boot_ci([r["metrics"][a_][m] - r["metrics"][b_][m] for r in subset])
                for m in ("f1_4_win", "f1_4_1to1", "f1_10_win", "f1_10_1to1")
            }

    print("SUMMARY:", json.dumps(summary, indent=2), flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(dict(summary=summary, rows=rows), open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
