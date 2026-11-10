"""E7b: identical weights, real SHD batches: compare spikes / gradients across kernels."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn as nn
from lifscan.shdnet import load, batch_dense, Net
T = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
torch.manual_seed(0)
ev, y = load("test", T)
net = Net("serial").cuda()
out = []
for b in range(5):
    idx = list(range(b * 64, b * 64 + 64))
    x = batch_dense(ev, idx, T); yy = y[idx].cuda()
    res = {}
    for mode in ("serial", "serial2", "scan", "sj"):
        net.set_mode("serial" if mode == "serial2" else mode)
        net.zero_grad()
        logits, (s1, s2) = net(x)
        nn.functional.cross_entropy(logits, yy).backward()
        res[mode] = (torch.cat([s1.flatten(), s2.flatten()]),
                     torch.cat([p.grad.flatten() for p in net.parameters()]).clone())
    ref_s, ref_g = res["serial"]
    row = {}
    for m in ("serial2", "scan", "sj"):
        s, gr = res[m]
        row[m] = dict(spikes_equal=bool(torch.equal(s, ref_s)), spike_mismatch=(s != ref_s).float().mean().item(),
                      grad_rel=((gr - ref_g).norm() / ref_g.norm()).item())
    out.append(row); print(b, json.dumps(row), flush=True)
json.dump(out, open(f"results/e7b_gradcheck_T{T}.json", "w"), indent=1)
