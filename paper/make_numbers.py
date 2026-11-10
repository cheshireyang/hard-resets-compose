"""Single producer of every experimental number quoted in main.tex.
Reads results/*.json, writes paper/numbers.tex and paper/tab_*.tex.  Fails loudly on missing input."""
import glob, json, os, statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "paper")


def load(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return json.load(open(p))


def fmt_x(v):
    return f"{v:.1f}" if v < 10 else f"{v:.0f}"


def sci(v):
    e = int(f"{v:e}".split("e")[1]); m = v / 10 ** e
    return f"{m:.1f}\\times10^{{{e}}}"


macros = {}
g9, g99 = load("e6c_fair_beta0.9.json"), load("e6c_fair_beta0.99.json")
for tag, rows in (("Nine", g9), ("NineNine", g99)):
    best_f = max(rows, key=lambda r: r["speedup_fwd"])
    best_fb = max(rows, key=lambda r: r["speedup_fb"])
    macros[f"FwdMax{tag}"] = fmt_x(best_f["speedup_fwd"])
    macros[f"FwdMax{tag}N"] = f"{best_f['N']:,}".replace(",", "{,}")
    macros[f"FwdMax{tag}T"] = f"{best_f['T']:,}".replace(",", "{,}")
    macros[f"FbMax{tag}"] = fmt_x(best_fb["speedup_fb"])
    macros[f"SJMax{tag}"] = fmt_x(max(r["speedup_fb_vs_sj"] for r in rows))
    wins = [r for r in rows if r["speedup_fwd"] > 1]
    macros[f"FwdWinMaxN{tag}"] = f"{max(r['N'] for r in wins):,}".replace(",", "{,}") if wins else "--"
    macros[f"MaxPieces{tag}"] = f"{max(r['scan_fwd_cfg']['pieces'] for r in rows):.1f}"
    macros[f"MaxOvf{tag}"] = f"{max(r['scan_fwd_cfg']['ovf'] for r in rows):.3f}"
    mf = max(r['scan_fwd_cfg']['flip'] for r in rows)
    macros[f"MaxFlip{tag}"] = "0" if mf == 0 else sci(mf)
    big = [r for r in rows if r["N"] >= 32768]
    macros[f"BigNFb{tag}"] = f"{min(r['speedup_fb'] for r in big):.2f}--{max(r['speedup_fb'] for r in big):.2f}"
macros["FwdMaxSpeedup"] = fmt_x(max(max(r["speedup_fwd"] for r in g9), max(r["speedup_fwd"] for r in g99)))
macros["FbMaxSpeedup"] = fmt_x(max(max(r["speedup_fb"] for r in g9), max(r["speedup_fb"] for r in g99)))
macros["SJMaxSpeedup"] = fmt_x(max(max(r["speedup_fb_vs_sj"] for r in g9), max(r["speedup_fb_vs_sj"] for r in g99)))
sj_ratio = [r["sj_fb"] / (r["serial_fwd"] + r["serial_bwd"]) for r in g9 if r["N"] <= 8192]
macros["SJvsSerialSmall"] = f"{min(sj_ratio):.1f}--{max(sj_ratio):.1f}"

# gradient checks on SHD batches
for T, W in ((1024, "Short"), (8192, "Long")):
    gc = load(f"e7b_gradcheck_T{T}.json")
    macros[f"GradRelScan{W}"] = sci(max(r["scan"]["grad_rel"] for r in gc))
    macros[f"GradRelSJ{W}"] = sci(max(r["sj"]["grad_rel"] for r in gc))
    macros[f"SpikesEqScan{W}"] = "all" if all(r["scan"]["spikes_equal"] for r in gc) else "NOT all"
    macros[f"SpikeMisSJ{W}"] = sci(max(r["sj"]["spike_mismatch"] for r in gc)) if any(
        r["sj"]["spike_mismatch"] > 0 for r in gc) else "0"

# SHD accuracy
acc = {}
for mode in ("serial", "scan", "sj"):
    fs = sorted(glob.glob(os.path.join(RES, f"e7_shd_{mode}_T1024_B64_s*.json")))
    finals = [100 * json.load(open(f))["epochs"][-1]["test_acc"] for f in fs
              if len(json.load(open(f))["epochs"]) == 30]
    if len(finals) < 3:
        raise RuntimeError(f"not enough SHD runs for {mode}: {len(finals)}")
    acc[mode] = finals
    macros[f"Acc{mode.capitalize() if mode != 'sj' else 'SJ'}"] = f"{st.mean(finals):.1f}\\pm{st.stdev(finals):.1f}"
    macros[f"NSeeds{mode.capitalize() if mode != 'sj' else 'SJ'}"] = str(len(finals))
n = min(len(acc["serial"]), len(acc["scan"]))
d = [a - b for a, b in zip(acc["scan"][:n], acc["serial"][:n])]
macros["AccDiffScanSerial"] = f"{st.mean(d):+.1f}\\pm{st.stdev(d):.1f}"
from scipy.stats import ttest_rel
macros["AccDiffP"] = f"{ttest_rel(acc['scan'][:n], acc['serial'][:n]).pvalue:.2f}"
for T in (8192,):
    for mode in ("serial", "scan", "sj"):
        f = os.path.join(RES, f"e7_shd_{mode}_T{T}_B16_s0.json")
        e = json.load(open(f))["epochs"]
        macros[f"AccLong{mode.capitalize() if mode != 'sj' else 'SJ'}"] = f"{100 * e[-1]['test_acc']:.1f}"

# no-repair behaviour on SHD batches
nr = load("e7d_norepair.json")
macros["NoRepairMismatch"] = "0" if max(r["mismatch_norepair"] for r in nr) == 0 else sci(max(r["mismatch_norepair"] for r in nr))
for T, W in ((1024, "Short"), (8192, "Long")):
    rr = [r["repair_rounds"] for r in nr if r["T"] == T]
    macros[f"RepairRounds{W}"] = f"{min(rr)}--{max(rr)}"
    macros[f"FireRate{W}"] = f"{100 * st.mean(r['rate'] for r in nr if r['T'] == T):.2f}"

# SHD per-iteration timing table
tm = load("e7c_timing.json")
cfgs = sorted({(r["T"], r["B"]) for r in tm})
lines = ["\\begin{tabular}{rrrrrrr}", "\\toprule",
         "$T$ & batch & $N$/layer & serial & scan & scan, no repair & SpikingJelly\\\\", "\\midrule"]
for T, B in cfgs:
    g = {r["mode"]: r["ms"] for r in tm if r["T"] == T and r["B"] == B}
    lines.append(f"{T} & {B} & {B * 256:,} & {g['serial']:.1f} & {g['scan']:.1f} & {g['scan_norepair']:.1f} & {g['sj']:.1f}\\\\".replace(",", "{,}"))
lines += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(OUT, "tab_shd_timing.tex"), "w").write("\n".join(lines) + "\n")

# appendix tables
e1 = load("e1_pieces.json")["random"]
lines = ["\\begin{tabular}{rrrrrrr}", "\\toprule",
         "$\\beta$ & $\\mu$ & $L$ & hard mean (max) & bound & soft mean (max) & bound\\\\", "\\midrule"]
for r in e1:
    if r["L"] not in (8, 32) or r["mu"] not in (0.05, 0.3):
        continue
    L = r["L"]
    lines.append(f"{r['beta']:g} & {r['mu']:g} & {L} & {r['hard_mean']:.1f} ({r['hard_max']}) & {L + 1} & "
                 f"{r['soft_mean']:.1f} ({r['soft_max']}) & {1 + L * (L + 1) // 2}\\\\")
lines += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(OUT, "tab_e1.tex"), "w").write("\n".join(lines) + "\n")
e2b = load("e2b_stress.json")
lines = ["\\begin{tabular}{rrrrr}", "\\toprule",
         "$\\beta$ & $\\varepsilon$ & flips scan32 vs serial32 & flips serial32 vs serial64 & repair rounds\\\\", "\\midrule"]
for r in e2b:
    lines.append(f"{r['beta']:g} & $10^{{{int(round(__import__('math').log10(r['eps'])))}}}$ & {r['flip_scan32_vs_ser32']:.0f} & "
                 f"{100 * r['flip_ser32_vs_ser64']:.2f}\\% & {r['repair_iters']}\\\\")
lines += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(OUT, "tab_stress.tex"), "w").write("\n".join(lines) + "\n")

with open(os.path.join(OUT, "numbers.tex"), "w") as f:
    f.write("% generated by make_numbers.py -- do not edit\n")
    for k, v in sorted(macros.items()):
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
for k, v in sorted(macros.items()):
    print(f"{k:28s} {v}")
