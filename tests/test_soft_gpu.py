import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from lifscan.fit import lif_soft
from lifscan import fast
torch.manual_seed(0)
T, N = 2048, 32
x = torch.randn(T, N, device="cuda"); W = torch.randn(T, N, device="cuda")
b = torch.tensor(0.93, device="cuda", requires_grad=True); g = torch.tensor(0.5, device="cuda", requires_grad=True)
S = lif_soft(g * x + 0.1, torch.zeros(N, device="cuda"), b); (W * S).sum().backward()
b2 = torch.tensor(0.93, dtype=torch.float64, device="cuda", requires_grad=True); g2 = torch.tensor(0.5, dtype=torch.float64, device="cuda", requires_grad=True)
u = torch.zeros(N, dtype=torch.float64, device="cuda"); ss = []
I2 = g2 * x.double() + 0.1
for t in range(T):
    v = b2 * u + I2[t]; s = fast.SpikeFn.apply(v - 1.0, 4.0); u = v - s.detach(); ss.append(s)
ss = torch.stack(ss); (W.double() * ss).sum().backward()
print("spikes equal:", torch.equal(S.detach(), ss.float().detach()), "dbeta", b.grad.item(), b2.grad.item(), "dgain", g.grad.item(), g2.grad.item())
