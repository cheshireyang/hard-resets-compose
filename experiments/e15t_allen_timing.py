"""E15t: per-iteration time of the hard-reset LIF fit on the 78 Allen cells of E15b/c, serial vs dispatching
kernels, with the tuned serial baselines (E26/E26b).  Same data preparation and loss as E15; 30 iterations each,
median of iterations 5..29.  usage: python3 experiments/e15t_allen_timing.py"""
import sys, os, json, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
sys.argv = [sys.argv[0], "0.2", "1", "1"]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "e15_allen_fit.py")).read().split("rows = []")[0])

ids = set(json.load(open("results/allen_cells_78.json")))
rows = []
for path in sorted(glob.glob("data/allen/*.npz")):
    meta, sw = load_cell(path)
    if meta["specimen"] not in ids or len(sw["Noise 1"]) < 2:
        continue
    Xtr, Str = stack(sw["Noise 1"])
    r = dict(cell=meta, T_train=Xtr.shape[0], N_train=Xtr.shape[1])
    for mode in ("serial", "auto"):
        _, _, times = fit("hard", Xtr, Str, 7, mode=mode, iters=30)
        r[f"hard_ms_{mode}"] = 1000 * float(np.median(times[5:]))
    rows.append(r)
    print(f"cell {meta['specimen']} T={r['T_train']} N={r['N_train']}: serial {r['hard_ms_serial']:.2f} ms, auto {r['hard_ms_auto']:.2f} ms "
          f"({r['hard_ms_serial'] / r['hard_ms_auto']:.1f}x)", flush=True)
    json.dump(rows, open("results/e15t_allen_timing.json", "w"), indent=1)
print("cells:", len(rows))
