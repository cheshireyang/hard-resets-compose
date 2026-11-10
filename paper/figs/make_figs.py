"""Figures for the paper.  python3 paper/figs/make_figs.py  (from project root)"""
import json, os, sys
from fractions import Fraction as Fr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from lifscan.exact import segment_map, pwa_segment, pwa_step_soft, serial_hard, serial_soft

OUT = os.path.join(ROOT, "paper", "figs")
RES = os.path.join(ROOT, "results")
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#8a8984", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "lines.linewidth": 1.5, "figure.dpi": 200, "savefig.bbox": "tight",
})


def fig_segment_map():
    """F(u0) for one random input segment, hard vs soft reset, exact."""
    import random
    random.seed(3)
    L, beta, th = 12, Fr(9, 10), Fr(1)
    I = [Fr(round(random.gauss(0.3, 0.3) * 100), 100) for _ in range(L)]
    xs = np.linspace(-10, 1.2, 4000)
    hard = [float(serial_hard(I, beta, th, Fr(0), Fr(x).limit_denominator(10 ** 6))[0][-1]) for x in xs]
    soft = [float(serial_soft(I, beta, th, Fr(x).limit_denominator(10 ** 6))[0][-1]) for x in xs]
    Mh = segment_map(I, beta, th, Fr(0)); Ms = pwa_segment([pwa_step_soft(x, beta, th) for x in I])
    fig, axs = plt.subplots(1, 2, figsize=(5.6, 1.9), sharey=False)
    for ax, ys, col, name, M, nb in ((axs[0], hard, BLUE, "hard reset", Mh, Mh.x),
                                     (axs[1], soft, ORANGE, "soft reset", Ms, Ms.bps)):
        ys = np.array(ys)
        jump = np.abs(np.diff(ys)) > 0.05
        ys_plot = ys.copy(); ys_plot[1:][jump] = np.nan
        ax.plot(xs, ys_plot, color=col)
        for b in nb:
            if xs[0] <= float(b) <= xs[-1]:
                ax.axvline(float(b), color=MUTED, lw=0.5, ls=(0, (2, 2)), zorder=0)
        ax.set_xlabel(r"initial state $u_0$")
        ax.set_title(f"{name}: {M.pieces} pieces (L = {L})", color=INK, fontsize=8, loc="left")
    axs[0].set_ylabel(r"$F(u_0)=u_L$")
    fig.savefig(os.path.join(OUT, "segment_map.pdf"))
    plt.close(fig)


def fig_pieces():
    e1 = json.load(open(os.path.join(RES, "e1_pieces.json")))["random"]
    e1b = json.load(open(os.path.join(RES, "e1b_scaling.json")))
    fig, axs = plt.subplots(1, 2, figsize=(5.6, 2.1))
    ax = axs[0]
    for beta, mu in ((0.99, 0.15),):
        rows = [r for r in e1 if r["beta"] == beta and r["mu"] == mu]
        L = np.array([r["L"] for r in rows])
        h = np.array([r["hard_mean"] for r in rows])
        sL = np.array([r["L"] for r in rows if r["soft_mean"] is not None])
        s = np.array([r["soft_mean"] for r in rows if r["soft_mean"] is not None])
        ax.plot(L, h, color=BLUE, marker="o", ms=3)
        ax.plot(sL, s, color=ORANGE, marker="o", ms=3)
        ax.plot(L, L + 1, color=BLUE, ls=(0, (3, 2)), lw=1)
        ax.plot(sL, 1 + sL * (sL + 1) / 2, color=ORANGE, ls=(0, (3, 2)), lw=1)
        ax.text(L[-1] * 1.05, h[-1], "hard", color=INK2, va="center")
        ax.text(sL[-1] * 1.05, s[-1], "soft", color=INK2, va="center")
        ax.text(L[-1] * 1.05, L[-1] + 1, "L+1", color=MUTED, va="center")
        ax.text(sL[-1] * 1.05, 1 + sL[-1] * (sL[-1] + 1) / 2, r"$1+\binom{L+1}{2}$", color=MUTED, va="center")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("segment length L"); ax.set_ylabel("pieces (mean of 20)")
    ax.set_title(r"(a) random inputs, $\beta=0.99$", loc="left", color=INK, fontsize=8)
    ax.set_xlim(3.5, 140)
    ax = axs[1]
    e1c = json.load(open(os.path.join(RES, "e1c_tight.json")))
    YELLOW = "#eda100"
    sel = ((0.9, 0.15, BLUE, r"$\beta$=0.9"), (0.99, 0.15, ORANGE, r"$\beta$=0.99"),
           (1.0, 0.15, AQUA, r"$\beta$=1"), (1.0, 0.0, YELLOW, r"$\beta$=1, $\mu$=0"))
    for beta, mu, col, lab in sel:
        rows = [r for r in e1c if r["beta"] == beta and r["mu"] == mu]
        L = np.array([r["L"] for r in rows])
        ax.plot(L, [r["tight_mean"] for r in rows], color=col, marker="o", ms=3, label=lab)
    rows = [r for r in e1c if r["beta"] == 0.99 and r["mu"] == 0.15]
    L = np.array([r["L"] for r in rows])
    ax.plot(L, [r["loose_mean"] for r in rows], color=ORANGE, ls=(0, (3, 2)), lw=1)
    ax.plot(L, [r["all_mean"] for r in rows], color=MUTED, ls=(0, (1, 1.5)), lw=1)
    ax.text(L[-1] * 1.15, rows[-1]["loose_mean"], "loose bound", color=MUTED, va="center", fontsize=7)
    ax.text(L[-1] * 1.15, rows[-1]["all_mean"], "no pruning", color=MUTED, va="center", fontsize=7)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("chunk length L"); ax.set_ylabel("pieces per chunk map")
    ax.set_title("(b) hard reset, pruned to reachable entry states", loc="left", color=INK, fontsize=8)
    ax.set_xlim(12, 2 ** 15)
    ax.legend(loc="upper left", fontsize=6.5, handlelength=1.5)
    fig.tight_layout(w_pad=2.5)
    fig.savefig(os.path.join(OUT, "pieces.pdf"))
    plt.close(fig)


def fig_gpu():
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    cmap = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", BLUE])
    fig, axs = plt.subplots(1, 3, figsize=(6.4, 2.0))
    panels = (("e6c_fair_beta0.9.json", "speedup_fwd", r"(a) exact forward, $\beta$=0.9"),
              ("e6c_fair_beta0.99.json", "speedup_fwd", r"(b) exact forward, $\beta$=0.99"),
              ("e6c_fair_beta0.9.json", "speedup_fb", r"(c) forward+backward, $\beta$=0.9"))
    for ax, (fn, key, title) in zip(axs, panels):
        path = os.path.join(RES, fn)
        if not os.path.exists(path):
            ax.set_axis_off(); continue
        rows = json.load(open(path))
        Ns = sorted({r["N"] for r in rows}); Ts = sorted({r["T"] for r in rows})
        Z = np.full((len(Ns), len(Ts)), np.nan)
        for r in rows:
            Z[Ns.index(r["N"]), Ts.index(r["T"])] = np.log2(r[key])
        ax.imshow(Z, cmap=cmap, norm=TwoSlopeNorm(0, -2.5, 5), aspect="auto", origin="lower")
        for i in range(len(Ns)):
            for j in range(len(Ts)):
                if not np.isnan(Z[i, j]):
                    v = 2 ** Z[i, j]
                    ax.text(j, i, f"{v:.1f}" if v < 10 else f"{v:.0f}", ha="center", va="center",
                            fontsize=6.5, color=INK)
        ax.set_xticks(range(len(Ts))); ax.set_xticklabels([f"{t // 1024}k" for t in Ts])
        ax.set_yticks(range(len(Ns))); ax.set_yticklabels([f"{n:,}" if n < 10 ** 4 else f"{n // 1024}k" for n in Ns])
        ax.set_xlabel("T"); ax.grid(False)
        ax.set_title(title, loc="left", color=INK, fontsize=8)
    axs[0].set_ylabel("N (batch x neurons)")
    fig.tight_layout(w_pad=1.0)
    fig.savefig(os.path.join(OUT, "gpu_speedup.pdf"))
    plt.close(fig)


def fig_shd():
    import glob
    fig, axs = plt.subplots(1, 2, figsize=(5.6, 2.0))
    found = False
    for mode, col, lab in (("serial", BLUE, "serial"), ("scan", ORANGE, "scan (ours)"), ("sj", AQUA, "SpikingJelly")):
        fs = sorted(glob.glob(os.path.join(RES, f"e7_shd_{mode}_T1024_B64_s*.json")))
        if not fs:
            continue
        found = True
        curves = [np.array([e["test_acc"] for e in json.load(open(f))["epochs"]]) for f in fs]
        n = min(len(c) for c in curves)
        C = np.stack([c[:n] for c in curves])
        ep = np.arange(1, n + 1)
        axs[0].plot(ep, C.mean(0), color=col, label=lab)
        axs[0].fill_between(ep, C.min(0), C.max(0), color=col, alpha=0.15, lw=0)
        f0 = os.path.join(RES, f"e7_shd_{mode}_T1024_B64_s0.json")
        if os.path.exists(f0):
            c0 = [e["train_acc"] for e in json.load(open(f0))["epochs"]]
            axs[1].plot(np.arange(1, len(c0) + 1), c0, color=col, label=lab)
    if found:
        axs[0].set_xlabel("epoch"); axs[0].set_ylabel("test accuracy")
        axs[0].set_title(f"(a) test accuracy, mean and range", loc="left", color=INK, fontsize=8)
        axs[1].set_xlabel("epoch"); axs[1].set_ylabel("train accuracy")
        axs[1].set_title("(b) seed 0, identical init", loc="left", color=INK, fontsize=8)
        axs[1].legend(loc="lower right")
        fig.tight_layout(w_pad=2.0)
        fig.savefig(os.path.join(OUT, "shd.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    which = sys.argv[1:] or ["segment", "pieces", "gpu", "shd"]
    if "segment" in which: fig_segment_map()
    if "pieces" in which: fig_pieces()
    if "gpu" in which: fig_gpu()
    if "shd" in which: fig_shd()
    print("ok")
