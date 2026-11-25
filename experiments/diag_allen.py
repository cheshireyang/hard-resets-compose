import sys, os, glob, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
sys.argv = ["x", "0.2", "1", "1"]
src = open("experiments/e15_allen_fit.py").read().split("rows = []")[0]
exec(src)
path = sorted(glob.glob("data/allen/*.npz"))[0]
meta, sw = load_cell(path); Xtr, Str = stack(sw["Noise 1"])
x = Xtr[:, 0].cpu().numpy()
print("x (100 pA): mean %.3f std %.3f min %.3f max %.3f frac>0.5 %.3f" % (x.mean(), x.std(), x.min(), x.max(), (x > 0.5).mean()))
print("target rate Hz", Str.mean().item() * 1000 / DT_MS)
for beta0, gain0, bias0 in ((0.97, 0.05, 0.0), (0.97, 0.1, 0.0), (0.99, 0.05, 0.0), (0.995, 0.02, 0.0), (0.97, 0.03, 0.0)):
    p = dict(zb=torch.tensor(math.log(beta0 / (1 - beta0)), device=dev), gain=torch.tensor(gain0, device=dev),
             bias=torch.tensor(bias0, device=dev), ur=torch.tensor(0.0, device=dev))
    S = simulate("hard", Xtr, p)
    print(beta0, gain0, "model rate Hz", S.mean().item() * 1000 / DT_MS, " F1", f1(S, Str, 20))
