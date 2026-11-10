"""E5: train the same 1-layer SNN with serial BPTT and with the scan operator; compare trajectories."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from lifscan.fast import *

def make_data(n, T=1024, Cin=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(0, 4, (n,), generator=g)
    rate = 0.02 + 0.0 * torch.zeros(n, T, Cin)
    for k in range(4):  # class k: burst in channel block k at time window k
        m = y == k
        rate[m, k * T // 4:(k + 1) * T // 4, k * Cin // 4:(k + 1) * Cin // 4] = 0.15
    return (torch.rand(n, T, Cin, generator=g) < rate).double(), y

def run(mode, steps=60, H=64, seed=0):
    torch.manual_seed(seed)
    W = (0.3 * torch.randn(32, H, dtype=torch.float64)).requires_grad_(True)
    R = (0.1 * torch.randn(H, 4, dtype=torch.float64)).requires_grad_(True)
    opt = torch.optim.Adam([W, R], lr=1e-2)
    X, y = make_data(64)
    hist = []; t0 = time.perf_counter()
    for it in range(steps):
        I = torch.einsum("ntc,ch->tnh", X, W).reshape(X.shape[1], -1)  # [T, n*H]
        u0 = torch.zeros(I.shape[1], dtype=torch.float64)
        if mode == "serial":
            u, s = serial_lif(I, 0.95, 1.0, 0.0, u0)
        else:
            u, s = scan_lif(I, 0.95, 1.0, 0.0, u0, C=32, K=64, repair=10**6)
        cnt = s.reshape(X.shape[1], X.shape[0], -1).mean(0)
        loss = torch.nn.functional.cross_entropy(cnt @ R * 10, y)
        opt.zero_grad(); loss.backward(); opt.step()
        hist.append(dict(loss=loss.item(), rate=s.mean().item(), acc=((cnt @ R).argmax(1) == y).double().mean().item()))
    return hist, time.perf_counter() - t0, W.detach()

hs, ts, Ws = run("serial")
hp, tp, Wp = run("scan")
dl = max(abs(a["loss"] - b["loss"]) for a, b in zip(hs, hp))
out = dict(serial_time=ts, scan_time=tp, max_loss_diff=dl, final_W_rel_diff=((Ws - Wp).norm() / Ws.norm()).item(),
           serial=hs, scan=hp)
print(f"serial {ts:.1f}s scan {tp:.1f}s | max |loss diff| {dl:.3e} | final W rel diff {out['final_W_rel_diff']:.3e}")
print("loss serial", [round(h['loss'], 4) for h in hs[::10]], "acc", hs[-1]["acc"])
print("loss scan  ", [round(h['loss'], 4) for h in hp[::10]], "acc", hp[-1]["acc"])
json.dump(out, open("results/e5_train.json", "w"), indent=1)
