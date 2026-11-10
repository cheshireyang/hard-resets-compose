"""E7: SHD, feed-forward 700-256-256-20 hard-reset LIF net, trained with serial / scan / sj.
usage: python3 experiments/e7_shd.py MODE T SEED [EPOCHS] [BATCH]"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn as nn
from lifscan.shdnet import load, batch_dense, Net

MODE, T, SEED = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
EPOCHS = int(sys.argv[4]) if len(sys.argv) > 4 else 30
BATCH = int(sys.argv[5]) if len(sys.argv) > 5 else 64
dev = "cuda"
DATASET = os.environ.get("DATASET", "shd")  # shd | ssc
NOUT = {"shd": 20, "ssc": 35}[DATASET]
PREFIX = "e7_shd" if DATASET == "shd" else f"e10_{DATASET}"
torch.manual_seed(SEED); np.random.seed(SEED)
tr_ev, tr_y = load("train", T, DATASET); te_ev, te_y = load("test", T, DATASET)
net = Net(MODE, n_out=NOUT).to(dev)
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
log = dict(dataset=DATASET, mode=MODE, T=T, seed=SEED, batch=BATCH, epochs=[])
g = torch.Generator().manual_seed(SEED)
for ep in range(EPOCHS):
    net.train(); perm = torch.randperm(len(tr_y), generator=g)
    t_c = 0.0; t0 = time.perf_counter(); tot = corr = 0; r1 = []; r2 = []
    for bi in range(0, len(perm) - BATCH + 1, BATCH):
        idx = perm[bi:bi + BATCH].tolist()
        x = batch_dense(tr_ev, idx, T); y = tr_y[idx].to(dev)
        torch.cuda.synchronize(); a = time.perf_counter()
        logits, (s1, s2) = net(x)
        loss = nn.functional.cross_entropy(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
        torch.cuda.synchronize(); t_c += time.perf_counter() - a
        corr += (logits.argmax(1) == y).sum().item(); tot += len(idx)
        r1.append(s1.mean().item()); r2.append(s2.mean().item())
    sched.step()
    ep_s = time.perf_counter() - t0
    net.eval(); tc = tt = 0
    with torch.no_grad():
        for bi in range(0, len(te_y), 128):
            idx = list(range(bi, min(bi + 128, len(te_y))))
            logits, _ = net(batch_dense(te_ev, idx, T))
            tc += (logits.argmax(1).cpu() == te_y[idx]).sum().item(); tt += len(idx)
    e = dict(ep=ep, train_acc=corr / tot, test_acc=tc / tt, compute_s=t_c, epoch_s=ep_s,
             rate1=float(np.mean(r1)), rate2=float(np.mean(r2)))
    log["epochs"].append(e); print(json.dumps(e), flush=True)
    json.dump(log, open(f"results/{PREFIX}_{MODE}_T{T}_B{BATCH}_s{SEED}.json", "w"), indent=1)
