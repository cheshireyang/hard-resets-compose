"""E2: float behaviour.  (a) scan(float32, no repair) vs serial(float32), compared with
serial(float32) vs serial(float64) as the 'inherent float noise' yardstick; (b) adversarial
dyadic inputs that hit the threshold exactly; (c) repair iterations needed."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from lifscan.fast import *
rows = []
T, N = 16384, 256
for beta in (0.5, 0.9, 0.99):
    for mu, sd in ((0.15, 0.4), (0.05, 0.1), (-0.1, 0.6)):
        torch.manual_seed(1)
        I64 = mu + sd * torch.randn(T, N, dtype=torch.float64); I32 = I64.float()
        u0 = torch.zeros(N)
        _, s64, _ = serial_forward_nograd(I64, beta, 1.0, 0.0, u0.double())
        _, s32, _ = serial_forward_nograd(I32, beta, 1.0, 0.0, u0)
        st = {}
        _, sc, _ = scan_forward_pruned(I32, beta, 1.0, 0.0, u0, C=128, K=64, stats=st)
        st2 = {}
        _, sc2, _ = scan_forward_pruned(I32, beta, 1.0, 0.0, u0, C=128, K=64, repair=10**6, stats=st2)
        first = lambda a, b: int(((a != b).any(1)).nonzero()[0].item()) if (a != b).any() else -1
        r = dict(beta=beta, mu=mu, sd=sd, rate=s32.mean().item(),
                 flip_scan_vs_ser32=(sc != s32).float().mean().item(),
                 neurons_diverged_scan=(sc != s32).any(0).float().mean().item(),
                 flip_ser32_vs_ser64=(s32 != s64.float()).float().mean().item(),
                 neurons_diverged_32v64=(s32 != s64.float()).any(0).float().mean().item(),
                 overflow=st["overflow_frac"], repair_iters=st2["repair_iters"],
                 bitexact_after_repair=bool(torch.equal(sc2, s32)))
        rows.append(r); print(r, flush=True)
# (b) adversarial: dyadic inputs, beta=1/2, theta=1 -> serial float is exact, many exact ties v==theta
torch.manual_seed(2)
I = (torch.randint(-4, 9, (4096, 512)).float() / 8)
u0 = torch.zeros(512)
_, s32, _ = serial_forward_nograd(I, 0.5, 1.0, 0.0, u0)
us, ss, vs = serial_forward_nograd(I, 0.5, 1.0, 0.0, u0)
ties = (vs == 1.0).float().mean().item()
st = {}
_, sc, _ = scan_forward_pruned(I, 0.5, 1.0, 0.0, u0, C=64, K=16, stats=st)
_, sc_full, _ = scan_forward(I, 0.5, 1.0, 0.0, u0, C=64)
adv = dict(tie_fraction=ties, flip_pruned=(sc != s32).float().mean().item(), flip_unpruned=(sc_full != s32).float().mean().item())
print("adversarial dyadic:", adv)
json.dump(dict(random=rows, adversarial=adv), open("results/e2_float.json", "w"), indent=1)
