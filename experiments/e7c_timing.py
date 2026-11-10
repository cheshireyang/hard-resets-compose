"""E7c: clean per-iteration timing of the SHD network (forward+backward of the two LIF layers
and linear maps), fixed random batches, for serial / scan / sj at several (T, batch)."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn as nn
from lifscan.shdnet import load, batch_dense, Net
rows = []
for T, B in ((1024, 64), (1024, 8), (4096, 16), (8192, 16), (8192, 4)):
    ev, y = load("test", T)
    x = batch_dense(ev, list(range(B)), T); yy = y[:B].cuda()
    for mode in ("serial", "scan", "scan_norepair", "sj"):
        torch.manual_seed(0); net = Net("scan" if mode.startswith("scan") else mode).cuda()
        if mode == "scan_norepair":
            net.l1.repair = net.l2.repair = 0
        def step():
            net.zero_grad(); logits, _ = net(x)
            nn.functional.cross_entropy(logits, yy).backward()
        for _ in range(3): step()
        torch.cuda.synchronize(); ts = []
        for _ in range(10):
            a = time.perf_counter(); step(); torch.cuda.synchronize(); ts.append(time.perf_counter() - a)
        ts.sort(); r = dict(T=T, B=B, mode=mode, ms=1000 * ts[len(ts) // 2]); rows.append(r); print(json.dumps(r), flush=True)
json.dump(rows, open("results/e7c_timing.json", "w"), indent=1)
