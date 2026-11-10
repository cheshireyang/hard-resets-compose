"""E11: does auto dispatch avoid the slow regime?  Forward+backward time of one LIF layer with
library defaults (no per-cell tuning): serial, scan (no repair), auto.  Also checks that all three
give identical spikes and matching gradients."""
import sys, os, json, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from lifscan import triton_lif as L

def timeit(fn, reps=25):
    fn(); torch.cuda.synchronize(); ts = []
    for _ in range(reps):
        a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    ts.sort(); return ts[len(ts) // 2]

rows = []
for N, T in itertools.product((128, 1024, 8192, 32768, 131072), (1024, 4096, 16384, 65536)):
    if N * T > 2 ** 28: continue
    torch.manual_seed(0)
    I0 = (0.15 + 0.4 * torch.randn(T, N, device="cuda")).contiguous(); u0 = torch.zeros(N, device="cuda")
    G = torch.randn(T, N, device="cuda")
    out = {}
    fns = {"serial": lambda I: L.lif_tuned(I, u0, beta=0.9, mode="serial"),
           "scan": lambda I: L.lif_tuned(I, u0, beta=0.9, mode="scan", repair=0),
           "auto": lambda I: L.lif_auto(I, u0, beta=0.9)}
    for name, f in fns.items():
        I = I0.clone().requires_grad_(True)
        def step():
            I.grad = None; s = f(I); (s * G).sum().backward()
        t = timeit(step)
        I.grad = None; s = f(I); (s * G).sum().backward()
        out[name] = (t, s.detach().clone(), I.grad.clone())
    r = dict(N=N, T=T, modes=L.auto_modes(N, T), serial=out["serial"][0], scan=out["scan"][0], auto=out["auto"][0])
    gs = out["serial"][2]
    r["spikes_equal"] = all(torch.equal(out[k][1], out["serial"][1]) for k in ("scan", "auto"))
    r["grad_rel_auto"] = ((out["auto"][2] - gs).norm() / gs.norm()).item()
    r["x_auto"] = r["serial"] / r["auto"]; r["x_scan"] = r["serial"] / r["scan"]
    rows.append(r)
    print(f"N={N:<7} T={T:<6} modes={r['modes']} serial {r['serial']:8.3f} scan {r['scan']:8.3f} (x{r['x_scan']:5.2f}) "
          f"auto {r['auto']:8.3f} (x{r['x_auto']:5.2f}) spikes_eq={r['spikes_equal']} grad_rel={r['grad_rel_auto']:.1e}", flush=True)
    json.dump(rows, open("results/e11_auto.json", "w"), indent=1)
    del I0, G, out; torch.cuda.empty_cache()
