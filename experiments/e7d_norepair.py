"""E7d: on real SHD batches, how often does the scan WITHOUT repair differ from the serial kernel,
and how many repair rounds does bit-exactness need?  Per layer, identical weights."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from lifscan.shdnet import load, batch_dense, Net
from lifscan import triton_lif as L
out = []
for T in (1024, 8192):
    ev, y = load("test", T)
    torch.manual_seed(0); net = Net("serial").cuda()
    for b in range(5):
        x = batch_dense(ev, list(range(b * 16, b * 16 + 16)), T)
        with torch.no_grad():
            I1 = net.l1.fc(x).reshape(T, -1).contiguous()
            u0 = torch.zeros(I1.shape[1], device="cuda")
            for li, I in enumerate((I1,)):
                _, Ss = L.serial_forward_tiled(I, 0.95, 1., 0., u0, BN=32, D=8, num_warps=1)
                st = {}
                _, Sn = L.scan_forward_v3(I, 0.95, 1., 0., u0, C=L._pick_C(T), K=16, repair=0, stats=st)
                st2 = {}
                _, Sr = L.scan_forward_v3(I, 0.95, 1., 0., u0, C=L._pick_C(T), K=16, repair=10 ** 6, stats=st2)
                r = dict(T=T, batch=b, layer=li + 1, rate=Ss.mean().item(), mismatch_norepair=(Sn != Ss).float().mean().item(),
                         neurons_affected=(Sn != Ss).any(0).float().mean().item(), repair_rounds=st2["repair_iters"],
                         exact_after=bool(torch.equal(Sr, Ss)), ovf=st["overflow_frac"])
                out.append(r); print(json.dumps(r), flush=True)
json.dump(out, open("results/e7d_norepair.json", "w"), indent=1)
