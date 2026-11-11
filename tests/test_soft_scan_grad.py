import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from lifscan.fit import lif_soft_scan
torch.manual_seed(0)
T, N = 16384, 8
x = torch.randn(T, N, device="cuda"); W = torch.randn(T, N, device="cuda")
res = {}
for mode in ("serial", "scan"):
    b = torch.tensor(0.95, device="cuda", requires_grad=True); g = torch.tensor(0.3, device="cuda", requires_grad=True)
    S = lif_soft_scan(g * x + 0.06, torch.zeros(N, device="cuda"), b, mode=mode); (W * S).sum().backward()
    res[mode] = (S.detach(), b.grad.item(), g.grad.item())
print("spikes equal:", torch.equal(res["serial"][0], res["scan"][0]),
      "dbeta", res["serial"][1], res["scan"][1], "dgain", res["serial"][2], res["scan"][2],
      "rel", abs(res["serial"][1] - res["scan"][1]) / abs(res["serial"][1]), abs(res["serial"][2] - res["scan"][2]) / abs(res["serial"][2]))
