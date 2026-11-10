"""SHD data + feed-forward hard-reset LIF network shared by E7 experiments."""
import h5py
import numpy as np
import torch
import torch.nn as nn
from . import triton_lif as L

import os
DATA = os.environ.get("SHD_DATA", "data/shd")  # directory with shd_train.h5 / shd_test.h5


def load(split, T):
    f = h5py.File(f"{DATA}/shd_{split}.h5", "r")
    y = torch.tensor(f["labels"][:].astype(np.int64))
    ev = []
    for tt, uu in zip(f["spikes"]["times"][:], f["spikes"]["units"][:]):
        b = np.minimum((tt * T).astype(np.int64), T - 1)  # 1 s window, T bins
        ev.append(torch.tensor(np.stack([b, uu.astype(np.int64)], 1)))
    return ev, y


def batch_dense(ev, idx, T, dev="cuda"):
    x = torch.zeros(T, len(idx), 700, device=dev)
    for j, i in enumerate(idx):
        e = ev[i].to(dev, non_blocking=True)
        x[e[:, 0], j, e[:, 1]] = 1.0
    return x


class LIFLayer(nn.Module):
    def __init__(self, din, dout, beta, mode, K=16, repair=10 ** 6):
        super().__init__()
        self.fc = nn.Linear(din, dout)
        nn.init.uniform_(self.fc.weight, -1.5 / din ** 0.5, 1.5 / din ** 0.5)
        self.beta, self.mode, self.K, self.repair = beta, mode, K, repair

    def forward(self, x):
        Tn, B, _ = x.shape
        I = self.fc(x).reshape(Tn, -1).contiguous()
        u0 = torch.zeros(I.shape[1], device=I.device)
        if self.mode in ("serial", "scan"):
            s = L.lif_tuned(I, u0, beta=self.beta, theta=1.0, u_r=0.0, alpha=4.0, mode=self.mode,
                            K=self.K, repair=self.repair)
        else:
            from spikingjelly.activation_based import functional as sjf, surrogate
            s, _, _ = sjf.lif_multi_step(I, u0, 1.0 / (1 - self.beta), False, 1.0, 0.0,
                                         surrogate.Sigmoid(alpha=4.0), True, False)
        return s.reshape(Tn, B, -1)


class Net(nn.Module):
    def __init__(self, mode, beta=0.95, H=256):
        super().__init__()
        self.l1 = LIFLayer(700, H, beta, mode)
        self.l2 = LIFLayer(H, H, beta, mode)
        self.out = nn.Linear(H, 20)

    def set_mode(self, mode):
        self.l1.mode = self.l2.mode = mode

    def forward(self, x):
        s1 = self.l1(x)
        s2 = self.l2(s1)
        return self.out(s2).mean(0) * 10, (s1, s2)
