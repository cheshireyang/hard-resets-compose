import sys, os, json, glob, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
_argv = sys.argv[:]; sys.argv = [sys.argv[0], "0.2", "1", "1"]
exec(open("experiments/e15_allen_fit.py").read().split("rows = []")[0])
from lifscan.triton_affine import lif_affine
from lifscan.fit import lif_soft_scan
BASE = {}
for f in glob.glob("results/e15b_allen_w*.json") + glob.glob("results/e15c_allen_w*.json"):
    for r in json.load(open(f)): BASE[r["cell"]["specimen"]] = r
n = 0
for path in sorted(glob.glob("data/allen/*.npz")):
    sid = int(os.path.basename(path)[:-4])
    if sid not in BASE: continue
    meta, sw = load_cell(path); Xte, Ste = stack(sw["Noise 2"]); b = BASE[sid]["models"]
    u0 = torch.zeros(Xte.shape[1], device=dev); win = int(WIN_MS / DT_MS)
    with torch.no_grad():
        Ih = b["hard"]["gain"] * Xte + b["hard"]["bias"]
        S1 = lif_param(Ih, u0, torch.tensor(b["hard"]["beta"], device=dev), torch.tensor(1.0, device=dev), torch.tensor(b["hard"]["ur"], device=dev), mode="serial")
        S2 = lif_affine(Ih, u0, torch.tensor(b["hard"]["beta"], device=dev), torch.tensor(0.0, device=dev), torch.tensor(b["hard"]["ur"], device=dev), mode="serial")
        Is = b["soft"]["gain"] * Xte + b["soft"]["bias"]
        S3 = lif_soft_scan(Is, u0, torch.tensor(b["soft"]["beta"], device=dev), mode="serial")
        S4 = lif_affine(Is, u0, torch.tensor(b["soft"]["beta"], device=dev), torch.tensor(1.0, device=dev), torch.tensor(-1.0, device=dev), mode="serial")
    print(sid, "stored hard F1 %.3f | lif_param %.3f | affine(g=0) %.3f || stored soft %.3f | lif_soft %.3f | affine(g=1) %.3f | rates %.4f %.4f" % (
        b["hard"]["f1_test"], f1(S1, Ste, win), f1(S2, Ste, win), b["soft"]["f1_test"], f1(S3, Ste, win), f1(S4, Ste, win), S1.mean().item(), S2.mean().item()))
    n += 1
    if n >= 5: break
