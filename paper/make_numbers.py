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

# SHD per-iteration timing table (second node, includes auto dispatch)
tm = load("e7c_timing_node2.json")
cfgs = sorted({(r["T"], r["B"]) for r in tm})
lines = ["\\begin{tabular}{rrrrrrrr}", "\\toprule",
         "$T$ & batch & $N$/layer & serial & scan & scan, no repair & auto & SpikingJelly\\\\", "\\midrule"]
for T, B in cfgs:
    g = {r["mode"]: r["ms"] for r in tm if r["T"] == T and r["B"] == B}
    lines.append(f"{T} & {B} & {B * 256:,} & {g['serial']:.1f} & {g['scan']:.1f} & {g['scan_norepair']:.1f} & {g['auto']:.1f} & {g['sj']:.1f}\\\\".replace(",", "{,}"))
    macros_key = f"ShdAuto{T}{B}"
lines += ["\\bottomrule", "\\end{tabular}"]
autox = [g_s / g_a for (g_s, g_a) in [({r["mode"]: r["ms"] for r in tm if r["T"] == T and r["B"] == B}["serial"],
                                       {r["mode"]: r["ms"] for r in tm if r["T"] == T and r["B"] == B}["auto"]) for T, B in cfgs]]
macros["ShdAutoMin"] = f"{min(autox):.2f}"; macros["ShdAutoMax"] = f"{max(autox):.1f}"
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

al = load("e8_alif_regions.json")
Ls = sorted({r["L"] for r in al}); models = ["LIF", "ALIF q=0.5", "ALIF q=1.0"]
lines = ["\\begin{tabular}{l" + "r" * len(Ls) + "}", "\\toprule", "model & " + " & ".join(f"$L={L}$" for L in Ls) + "\\\\", "\\midrule"]
for m in models:
    cells = []
    for L in Ls:
        r = [x for x in al if x["model"] == m and x["L"] == L][0]
        cells.append(f"{r['mean']:.1f} ({r['max']})")
    lines.append(m.replace("q=", "$q=$") + " & " + " & ".join(cells) + "\\\\")
lines += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(OUT, "tab_alif.tex"), "w").write("\n".join(lines) + "\n")

# E9: long sequences, two- vs three-level (second RTX 5090 node)
lines = ["\\begin{tabular}{rrrrrrrr}", "\\toprule",
         "$\\beta$ & $N$ & $T$ & serial fwd (ms) & two-level & three-level & bwd $\\times$ & fwd+bwd $\\times$\\\\", "\\midrule"]
e9max = {}
for b in ("0.9", "0.99"):
    rows9 = load(f"e9_threelevel_beta{b}.json")
    for r in rows9:
        lines.append(f"{b} & {r['N']:,} & {r['T']:,} & {r['serial_fwd']:.2f} & {r['x_v3']:.1f}$\\times$ & {r['x_v4']:.1f}$\\times$ & "
                     f"{r['serial_bwd'] / r['scan_bwd']:.0f}$\\times$ & {r['x_fb_v4']:.0f}$\\times$\\\\".replace(",", "{,}"))
    e9max[b] = max(rows9, key=lambda r: r["x_v3"])
    macros["LongFwdMax" + ("Nine" if b == "0.9" else "NineNine")] = fmt_x(e9max[b]["x_v3"])
    macros["LongFbMax" + ("Nine" if b == "0.9" else "NineNine")] = fmt_x(max(r["x_fb_v4"] for r in rows9))
    macros["LongVFourBest" + ("Nine" if b == "0.9" else "NineNine")] = str(sum(r["x_v4"] > r["x_v3"] for r in rows9))
    macros["LongCells" + ("Nine" if b == "0.9" else "NineNine")] = str(len(rows9))
    macros["LongMaxFlip" + ("Nine" if b == "0.9" else "NineNine")] = "0" if max(max(r["v3_cfg"]["flip"], r["v4_cfg"]["flip"]) for r in rows9) == 0 else "nonzero"
lines += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(OUT, "tab_long.tex"), "w").write("\n".join(lines) + "\n")

# E10: SSC (optional until runs finish)
ssc = {}
for mode in ("serial", "scan", "sj"):
    fs = sorted(glob.glob(os.path.join(RES, f"e10_ssc_{mode}_T1024_B64_s*.json")))
    fin = [100 * json.load(open(f))["epochs"][-1]["test_acc"] for f in fs if len(json.load(open(f))["epochs"]) == 20]
    ssc[mode] = fin
if all(len(v) >= 2 for v in ssc.values()):
    for mode, key in (("serial", "Serial"), ("scan", "Scan"), ("sj", "SJ")):
        v = ssc[mode]
        macros[f"SscAcc{key}"] = f"{st.mean(v):.1f}\\pm{st.stdev(v):.1f}"
        macros[f"SscN{key}"] = str(len(v))
    gcs = load("e10_ssc_gradcheck_T1024.json")
    macros["SscGradRelScan"] = sci(max(r["scan"]["grad_rel"] for r in gcs))
    macros["SscSpikesEqScan"] = "all" if all(r["scan"]["spikes_equal"] for r in gcs) else "NOT all"
    macros["SscGradRelSJ"] = sci(max(r["sj"]["grad_rel"] for r in gcs))
    n = min(len(ssc["scan"]), len(ssc["serial"]))
    d = [a - b for a, b in zip(ssc["scan"][:n], ssc["serial"][:n])]
    macros["SscDiff"] = f"{st.mean(d):+.1f}"
else:
    print("WARNING: SSC results incomplete", {k: len(v) for k, v in ssc.items()})

# E11: auto dispatch (single layer, library defaults)
au = load("e11_auto.json")
lines = ["\\begin{tabular}{rrlrrr}", "\\toprule",
         "$N$ & $T$ & auto choice (fwd/bwd) & serial (ms) & scan $\\times$ & auto $\\times$\\\\", "\\midrule"]
for r in au:
    lines.append(f"{r['N']:,} & {r['T']:,} & {r['modes'][0]}/{r['modes'][1]} & {r['serial']:.2f} & {r['x_scan']:.2f} & {r['x_auto']:.2f}\\\\".replace(",", "{,}"))
lines += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(OUT, "tab_auto.tex"), "w").write("\n".join(lines) + "\n")
macros["AutoMin"] = f"{min(r['x_auto'] for r in au):.2f}"
macros["AutoMax"] = fmt_x(max(r["x_auto"] for r in au))
macros["AutoScanMin"] = f"{min(r['x_scan'] for r in au):.2f}"
macros["AutoCells"] = str(len(au))
macros["AutoSpikesEq"] = "all" if all(r["spikes_equal"] for r in au) else "NOT all"
_g = max(r["grad_rel_auto"] for r in au)
macros["AutoGradRel"] = "0" if _g == 0 else sci(_g)

# E12: ALIF exact post-reset maps
e12 = load("e12_alif_exact.json")
lines = ["\\begin{tabular}{lrrrrrr}", "\\toprule",
         "$(\\beta,\\rho,q)$ & $L=8$ & $L=16$ & $L=32$ & $L=64$ & $L=128$ & bound, $L=64$\\\\", "\\midrule"]
groups = {}
for r in e12["rows"]:
    groups.setdefault((r["beta"], r["rho"], r["q"]), {})[r["L"]] = r
for (b, rh, qq), d in groups.items():
    cells = [f"{d[L]['mean']:.1f} ({d[L]['max']})" for L in (8, 16, 32, 64, 128)]
    lines.append(f"({b:g}, {rh:g}, {qq:g}) & " + " & ".join(cells) + f" & {d[64]['bound2d_mean']:.0f}\\\\")
lines += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(OUT, "tab_alif_exact.tex"), "w").write("\n".join(lines) + "\n")
macros["AlifMonoChecks"] = f"{e12['mono_checks']:,}".replace(",", "{,}")
macros["AlifMonoViol"] = str(e12["mono_violations"])

# E13: reset-rule classes
e13 = load("e13_reset_classes.json")
rules = ["hard (r=0)", "soft (r=v-1)", "proportional (r=v/2), I>=0", "proportional (r=v/2), any I", "affine (r=v/2-1/2)", "decreasing (r=-v/2)"]
names = {"hard (r=0)": "constant, $r=0$", "soft (r=v-1)": "soft, $r(v)=v-1$", "proportional (r=v/2), I>=0": "proportional, $r(v)=v/2$, $I_t\\ge0$",
         "proportional (r=v/2), any I": "proportional, $r(v)=v/2$, any $I_t$", "affine (r=v/2-1/2)": "affine, $r(v)=(v-1)/2$", "decreasing (r=-v/2)": "decreasing, $r(v)=-v/2$ (hill-climbing)"}
Ls = sorted({r["L"] for r in e13})
lines = ["\\begin{tabular}{l" + "r" * len(Ls) + "}", "\\toprule", "reset rule & " + " & ".join(f"$L={L}$" for L in Ls) + "\\\\", "\\midrule"]
for ru in rules:
    lines.append(names[ru] + " & " + " & ".join(str([r for r in e13 if r["rule"] == ru and r["L"] == L][0]["max"]) for L in Ls) + "\\\\")
# E23: explicit constructions of Theorem 7(c) and the expansion examples (all exact, all checked against the serial recurrence)
e23 = load("e23_reset_lower_bounds.json")
assert all(r["serial_ok"] is not False for r in e23)
rev = [r for r in e23 if r["part"] == "reversing"]
assert all(r["pieces"] == 2 ** r["L"] and r["serial_ok"] for r in rev)
def rev_row(beta, gamma, delta):
    return " & ".join(str([r for r in rev if (r["beta"], r["gamma"], r["delta"]) == (beta, gamma, delta) and r["L"] == L][0]["pieces"]) for L in Ls)
lines.append("reflecting, $r(v)=2-v$ (construction) & " + rev_row("9/10", "1", "0") + "\\\\")
lines.append("decreasing, $r(v)=-v/2$ (construction) & " + rev_row("1/2", "1/2", "3/2") + "\\\\")
lines.append("\\midrule")
lines.append("$L+1$ & " + " & ".join(str(L + 1) for L in Ls) + "\\\\")
lines.append("$1+L(L+1)/2$ & " + " & ".join(str(1 + L * (L + 1) // 2) for L in Ls) + "\\\\")
lines.append("$2^L$ & " + " & ".join(str(2 ** L) for L in Ls) + "\\\\")
lines += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(OUT, "tab_resets.tex"), "w").write("\n".join(lines) + "\n")
macros["RevConfigs"] = str(len({(r["beta"], r["gamma"], r["delta"]) for r in rev})); macros["RevLmax"] = str(max(r["L"] for r in rev))
dec = [r for r in e13 if r["rule"] == "decreasing (r=-v/2)"]
assert all(r["max"] <= 1 + r["L"] * (r["L"] + 1) // 2 for r in dec)       # bounded-input hill-climbing stays below the quadratic bound
macros["DecHillMax"] = str(max(r["max"] for r in dec)); macros["DecHillL"] = str(max(r["L"] for r in dec))
def first_above(rows_):
    return next(r["L"] for r in sorted(rows_, key=lambda r: r["L"]) if r["pieces"] > r["quad"])
exb = lambda bb: [r for r in e23 if r["part"] == "expanding" and r["beta"] == bb]
macros["ExpTwoFirst"] = str(first_above(exb("2"))); macros["ExpThreeHalvesFirst"] = str(first_above(exb("3/2")))
assert all(r["pieces"] == 2 ** r["L"] for r in exb("2"))
last = max(exb("3/2"), key=lambda r: r["L"])
macros["ExpThreeHalvesL"] = str(last["L"]); macros["ExpThreeHalvesPieces"] = f"{last['pieces']:,}".replace(",", "{,}"); macros["ExpThreeHalvesQuad"] = str(last["quad"])
exg = [r for r in e23 if r["part"] == "expanding_reset" and r["gamma"] == "2"]
assert all(r["pieces"] >= r["lemma_lower"] for r in e23 if r["part"] == "expanding_reset")
macros["GamTwoFirst"] = str(first_above(exg))
lk = [r for r in e23 if r["part"] == "expanding_reset_leaky"]
macros["GamLeakyFirst"] = str(first_above(lk)); r0_ = [r for r in lk if r["L"] == first_above(lk)][0]
macros["GamLeakyPieces"] = str(r0_["pieces"]); macros["GamLeakyQuad"] = str(r0_["quad"])

# E14: parameter recovery (CPU accuracy comparison + GPU timing)
cpu = load("e14_fit_cpu_T8192_N8.json")
def agg(kind):
    rr = [r for r in cpu["runs"] if r["kind"] == kind]
    return {k: max(r["rel_err"][k] for r in rr) for k in ("beta", "gain", "bias")}, min(r["f1_own"] for r in rr), max(r["f1_own"] for r in rr)
lines = ["\\begin{tabular}{lrrrr}", "\\toprule", "fitted model & $\\beta$ err. & gain err. & bias err. & held-out F1\\\\", "\\midrule"]
for kind, nm in (("hard", "hard reset (exact)"), ("soft", "soft reset"), ("noreset", "no reset")):
    e, lo, hi = agg(kind)
    f1s = f"{lo:.3f}" if abs(hi - lo) < 5e-4 else f"{lo:.3f}--{hi:.3f}"
    pct = lambda v: f"{100 * v:.0f}" if 100 * v >= 10 else f"{100 * v:.2g}"
    lines.append(f"{nm} & {pct(e['beta'])}\\% & {pct(e['gain'])}\\% & {pct(e['bias'])}\\% & {f1s}\\\\")
lines += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(OUT, "tab_fit_synth.tex"), "w").write("\n".join(lines) + "\n")
gpu_rows = []
for T in (65536, 262144, 1048576):
    d = load(f"e14_fit_cuda_T{T}_N16.json")
    ser = [r for r in d["runs"] if r["kind"] == "hard:serial"]; au = [r for r in d["runs"] if r["kind"] == "hard:auto"]
    same = all(abs(a["fit"][k] - b["fit"][k]) < 1e-6 for a, b in zip(ser, au) for k in ("beta", "gain", "bias"))
    gpu_rows.append((T, ser[0]["ms_per_iter"], au[0]["ms_per_iter"], same, max(max(r["rel_err"].values()) for r in au), d["iters"]))
macros["FitSpeedMin"] = f"{min(r[1] / r[2] for r in gpu_rows):.1f}"
macros["FitSpeedMax"] = f"{max(r[1] / r[2] for r in gpu_rows):.1f}"
macros["FitSameAll"] = "identical" if all(r[3] for r in gpu_rows) else "NOT identical"
macros["FitMsSerialMillion"] = f"{gpu_rows[-1][1]:.0f}"; macros["FitMsAutoMillion"] = f"{gpu_rows[-1][2]:.0f}"
macros["FitMsSerialShort"] = f"{gpu_rows[0][1]:.1f}"; macros["FitMsAutoShort"] = f"{gpu_rows[0][2]:.1f}"
macros["FitHardErrMax"] = sci(max(max(r["rel_err"].values()) for r in cpu["runs"] if r["kind"] == "hard"))

# E15: Allen recordings
al = load("e15_allen_fit_dt0.2.json")
lines = ["\\begin{tabular}{llrrrrrr}", "\\toprule",
         "cell & type & rate (Hz) & reliability & hard & soft & no reset & ms/iter serial / auto\\\\", "\\midrule"]
for r in al:
    m = r["models"]
    lines.append(f"{r['cell']['specimen']} & {r['cell']['dendrite']} & {r['rate_hz']:.1f} & {r['reliability']:.2f} & "
                 f"{m['hard']['f1_test']:.3f} & {m['soft']['f1_test']:.3f} & {m['noreset']['f1_test']:.3f} & "
                 f"{r['hard_ms_serial']:.1f} / {r['hard_ms_auto']:.1f}\\\\")
lines += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(OUT, "tab_allen.tex"), "w").write("\n".join(lines) + "\n")
hf = [r["models"]["hard"]["f1_test"] for r in al]; sf = [r["models"]["soft"]["f1_test"] for r in al]
nf = [r["models"]["noreset"]["f1_test"] for r in al]; rl = [r["reliability"] for r in al]
from scipy.stats import wilcoxon
macros["AllenCells"] = str(len(al))
macros["AllenFHard"] = f"{st.mean(hf):.3f}"; macros["AllenFSoft"] = f"{st.mean(sf):.3f}"; macros["AllenFNo"] = f"{st.mean(nf):.3f}"
macros["AllenRel"] = f"{st.mean(rl):.3f}"
macros["AllenHardWins"] = str(sum(h > s for h, s in zip(hf, sf)))
macros["AllenDiff"] = f"{st.mean([h - s for h, s in zip(hf, sf)]):+.3f}"
macros["AllenP"] = f"{wilcoxon(hf, sf).pvalue:.2g}" if len(al) >= 6 else "n/a"
macros["AllenFracRel"] = f"{100 * st.mean([h / r for h, r in zip(hf, rl)]):.0f}"
sp = [r["hard_ms_serial"] / r["hard_ms_auto"] for r in al]
macros["AllenSpeed"] = f"{st.mean(sp):.1f}"
macros["AllenT"] = f"{al[0]['T_train']:,}".replace(",", "{,}")

# E15b/c: larger, balanced Allen experiment (merged workers; timing comes from the e15 v1 run above)
merged = []
for f in sorted(glob.glob(os.path.join(RES, "e15b_allen_w*.json")) + glob.glob(os.path.join(RES, "e15c_allen_w*.json"))):
    merged += json.load(open(f))
seen = set(); big = []
for r in merged:
    if r["cell"]["specimen"] not in seen:
        seen.add(r["cell"]["specimen"]); big.append(r)
if len(big) >= 20:
    grp = lambda r: "spiny" if r["cell"]["dendrite"] == "spiny" else "aspiny/sparsely spiny"
    lines = ["\\begin{tabular}{lrrrrrrr}", "\\toprule",
             "cells & $n$ & reliability & hard & soft & no reset & hard $>$ soft & Wilcoxon $p$\\\\", "\\midrule"]
    for gname, sel in (("all", big), ("spiny", [r for r in big if grp(r) == "spiny"]),
                       ("aspiny / sparsely spiny", [r for r in big if grp(r) != "spiny"])):
        h = [r["models"]["hard"]["f1_test"] for r in sel]; so = [r["models"]["soft"]["f1_test"] for r in sel]
        no = [r["models"]["noreset"]["f1_test"] for r in sel]; rl = [r["reliability"] for r in sel]
        pv = wilcoxon(h, so).pvalue if len(sel) >= 6 else float("nan")
        lines.append(f"{gname} & {len(sel)} & {st.mean(rl):.3f} & {st.mean(h):.3f} & {st.mean(so):.3f} & {st.mean(no):.3f} & "
                     f"{sum(a > b for a, b in zip(h, so))}/{len(sel)} & {pv:.2g}\\\\")
        key = {"all": "All", "spiny": "Spiny", "aspiny / sparsely spiny": "Aspiny"}[gname]
        macros[f"Big{key}N"] = str(len(sel)); macros[f"Big{key}Hard"] = f"{st.mean(h):.3f}"; macros[f"Big{key}Soft"] = f"{st.mean(so):.3f}"
        macros[f"Big{key}No"] = f"{st.mean(no):.3f}"; macros[f"Big{key}Rel"] = f"{st.mean(rl):.3f}"
        macros[f"Big{key}Wins"] = str(sum(a > b for a, b in zip(h, so))); macros[f"Big{key}P"] = f"{pv:.2g}"
        macros[f"Big{key}Diff"] = f"{st.mean([a - b for a, b in zip(h, so)]):+.3f}"
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(OUT, "tab_allen_summary.tex"), "w").write("\n".join(lines) + "\n")
    # per-cell appendix tables, 40 rows each
    big.sort(key=lambda r: (grp(r), r["cell"]["specimen"]))
    chunks = [big[i:i + 40] for i in range(0, len(big), 40)]
    for ci, ch in enumerate(chunks):
        lines = ["\\begin{tabular}{llrrrrr}", "\\toprule", "cell & type & rate (Hz) & reliability & hard & soft & no reset\\\\", "\\midrule"]
        for r in ch:
            m = r["models"]
            lines.append(f"{r['cell']['specimen']} & {r['cell']['dendrite']} & {r['rate_hz']:.1f} & {r['reliability']:.2f} & "
                         f"{m['hard']['f1_test']:.3f} & {m['soft']['f1_test']:.3f} & {m['noreset']['f1_test']:.3f}\\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        open(os.path.join(OUT, f"tab_allen_cells{ci + 1}.tex"), "w").write("\n".join(lines) + "\n")
    macros["BigChunks"] = str(len(chunks))
else:
    print("WARNING: balanced Allen run incomplete:", len(big))

# E17/E18: soft-reset scan
e17 = load("e17_soft_bench.json")
lines = ["\\begin{tabular}{rrrrrrr}", "\\toprule", "$\\beta$ & $N$ & $T$ & serial (ms) & soft scan (ms) & speedup & pieces\\\\", "\\midrule"]
for r in e17:
    lines.append(f"{r['beta']:g} & {r['N']:,} & {r['T']:,} & {r['serial']:.2f} & {r['scan']:.2f} & {r['speedup']:.2f}$\\times$ & {r['cfg']['pieces']:.1f}\\\\".replace(",", "{,}"))
lines += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(OUT, "tab_soft.tex"), "w").write("\n".join(lines) + "\n")
macros["SoftMax"] = f"{max(r['speedup'] for r in e17):.1f}"
best = max(e17, key=lambda r: r["speedup"])
macros["SoftMaxN"] = str(best["N"]); macros["SoftMaxT"] = f"{best['T']:,}".replace(",", "{,}"); macros["SoftMaxBeta"] = f"{best['beta']:g}"
macros["SoftMaxFlip"] = "0" if max(r["cfg"]["flip"] for r in e17) == 0 else sci(max(r["cfg"]["flip"] for r in e17))
e18_all = load("e18_soft_allen.json")
e18 = [r for r in e18_all if r["max_param_rel_diff"] == r["max_param_rel_diff"]]   # drop fits that diverged (NaN) in both modes
macros["SoftAllenTested"] = str(len(e18_all)); macros["SoftAllenDiverged"] = str(len(e18_all) - len(e18))
macros["SoftAllenCells"] = str(len(e18))
macros["SoftAllenSpeed"] = f"{st.mean(r['speedup'] for r in e18):.1f}"
macros["SoftAllenSpeedMin"] = f"{min(r['speedup'] for r in e18):.1f}"; macros["SoftAllenSpeedMax"] = f"{max(r['speedup'] for r in e18):.1f}"
macros["SoftAllenParamDiff"] = sci(max(max(r["max_param_rel_diff"] for r in e18), 1e-12))
macros["SoftAllenFOneDiff"] = f"{max(abs(r['serial_f1'] - r['scan_f1']) for r in e18):.3f}"
macros["SoftAllenMsSerial"] = f"{st.mean(r['serial_ms'] for r in e18):.0f}"; macros["SoftAllenMsScan"] = f"{st.mean(r['scan_ms'] for r in e18):.0f}"

# E19: exact Picard iteration baseline
pc = load("e19_picard.json")
lines = ["\\begin{tabular}{rrrrrrrr}", "\\toprule",
         "$N$ & $T$ & rate & iterations & error after 3 & Picard (ms) & serial (ms) & scan (ms)\\\\", "\\midrule"]
for r in pc:
    lines.append(f"{r['N']:,} & {r['T']:,} & {r['rate']:.3f} & {r['iters']}{'' if r['exact'] else '$^*$'} & {100 * r['err_after3']:.0f}\\% & "
                 f"{r['ms_picard']:.0f} & {r['ms_serial']:.2f} & {r['ms_scan']:.2f}\\\\".replace(",", "{,}"))
lines += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(OUT, "tab_picard.tex"), "w").write("\n".join(lines) + "\n")
macros["PicItMin"] = str(min(r["iters"] for r in pc)); macros["PicItMax"] = str(max(r["iters"] for r in pc))
macros["PicErrMin"] = f"{100 * min(r['err_after3'] for r in pc):.0f}"; macros["PicErrMax"] = f"{100 * max(r['err_after3'] for r in pc):.0f}"
ratio = [r["ms_picard"] / r["ms_scan"] for r in pc]
macros["PicSlowMin"] = f"{min(ratio):.0f}"; macros["PicSlowMax"] = f"{max(ratio):.0f}"
macros["PicInexact"] = str(sum(not r["exact"] for r in pc)); macros["PicCells"] = str(len(pc))

# E20: FPT / PSN head-to-head
e20 = load("e20_fpt_psn.json")
lines = ["\\begin{tabular}{rrrrrrrrr}", "\\toprule",
         "$\\lambda$ & $N$ & $T$ & soft scan & soft serial & FPT & PSN & FPT spikes wrong & scan spikes wrong\\\\", "\\midrule"]
for r in e20:
    f = lambda v: "--" if v is None else f"{v:.2f}"
    lines.append(f"{r['lam']:g} & {r['N']:,} & {r['T']:,} & {r['ms_soft_scan']:.2f} & {r['ms_soft_serial']:.2f} & {f(r.get('ms_fpt'))} & {f(r.get('ms_psn'))} & "
                 f"{100 * r['mismatch_fpt_bernoulli']:.1f}\\% & {100 * r['mismatch_scan']:.1f}\\%\\\\".replace(",", "{,}"))
lines += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(OUT, "tab_fpt.tex"), "w").write("\n".join(lines) + "\n")
f5 = [r for r in e20 if r["lam"] == 0.5]; f9 = [r for r in e20 if r["lam"] == 0.9]
macros["FptErrHalf"] = f"{100 * min(r['mismatch_fpt_bernoulli'] for r in f5):.1f}--{100 * max(r['mismatch_fpt_bernoulli'] for r in f5):.1f}"
macros["FptErrNine"] = f"{100 * min(r['mismatch_fpt_bernoulli'] for r in f9):.0f}--{100 * max(r['mismatch_fpt_bernoulli'] for r in f9):.0f}"
macros["FptDetErrNine"] = f"{100 * max(r['mismatch_fpt_det'] for r in f9):.0f}"
macros["FptRateNine"] = f"{100 * max(r['rate_fpt_det'] for r in f9):.0f}"; macros["ExactRateNine"] = f"{100 * max(r['rate_exact'] for r in f9):.0f}"
big = [r for r in e20 if r["T"] == 32768 and r["N"] == 128 and r["lam"] == 0.9][0]
macros["FptMsLong"] = f"{big['ms_fpt']:.1f}"; macros["ScanMsLong"] = f"{big['ms_soft_scan']:.1f}"; macros["PsnMsLong"] = f"{big['ms_psn']:.1f}"
macros["ScanErrAll"] = "0" if max(r["mismatch_scan"] for r in e20) == 0 else sci(max(r["mismatch_scan"] for r in e20))

# E21: chunk-restart (Bullet-Trains-style, discrete) vs firing rate
e21 = load("e21_chunk_restart.json")
lines = ["\\begin{tabular}{rrrrr}", "\\toprule", "$N$ & rate & chunk restart (ms) & serial (ms) & map scan (ms)\\\\", "\\midrule"]
for r in e21:
    lines.append(f"{r['N']:,} & {100 * r['rate']:.1f}\\% & {r['ms_chunk_restart']:.2f} & {r['ms_serial']:.2f} & {r['ms_scan']:.2f}\\\\".replace(",", "{,}"))
lines += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(OUT, "tab_chunkrestart.tex"), "w").write("\n".join(lines) + "\n")
c128 = [r for r in e21 if r["N"] == 128]
macros["CrLow"] = f"{min(r['ms_chunk_restart'] for r in c128):.2f}"; macros["CrHigh"] = f"{max(r['ms_chunk_restart'] for r in c128):.2f}"
macros["ScanFlatLow"] = f"{min(r['ms_scan'] for r in c128):.2f}"; macros["ScanFlatHigh"] = f"{max(r['ms_scan'] for r in c128):.2f}"
macros["CrRateLow"] = f"{100 * min(r['rate'] for r in c128):.0f}"; macros["CrRateHigh"] = f"{100 * max(r['rate'] for r in c128):.0f}"

# E22: learned reset (negative result)
warm = []
for f_ in glob.glob(os.path.join(RES, "e22_allen_affine_warm_w*.json")): warm += json.load(open(f_))
basemap = {}
for f_ in glob.glob(os.path.join(RES, "e15b_allen_w*.json")) + glob.glob(os.path.join(RES, "e15c_allen_w*.json")):
    for r in json.load(open(f_)): basemap[r["cell"]["specimen"]] = r
pairs = [(a, basemap[a["cell"]["specimen"]]) for a in warm if not a.get("failed") and a["cell"]["specimen"] in basemap]
macros["AffCells"] = str(len(pairs)); macros["AffFailed"] = str(sum(1 for a in warm if a.get("failed")))
macros["AffLossLower"] = str(sum(a["train_loss"] < min(b["models"]["hard"]["train_loss"], b["models"]["soft"]["train_loss"]) for a, b in pairs))
macros["AffFOne"] = f"{st.mean(a['f1_test'] for a, b in pairs):.3f}"
macros["AffHardF"] = f"{st.mean(b['models']['hard']['f1_test'] for a, b in pairs):.3f}"
macros["AffSoftF"] = f"{st.mean(b['models']['soft']['f1_test'] for a, b in pairs):.3f}"
macros["AffBestOf"] = f"{st.mean(max(b['models']['hard']['f1_test'], b['models']['soft']['f1_test']) for a, b in pairs):.3f}"

cold = []
for f_ in glob.glob(os.path.join(RES, "e22_allen_affine_cold_w*.json")): cold += json.load(open(f_))
cok = [r for r in cold if not r.get("failed") and r.get("ms_serial")]
ratios = sorted(r["ms_serial"] / r["ms_scan"] for r in cok)
macros["AffSpeed"] = f"{st.median(ratios):.1f}"; macros["AffSpeedMin"] = f"{ratios[0]:.1f}"; macros["AffSpeedMax"] = f"{ratios[-1]:.1f}"
macros["AffColdFOne"] = f"{st.mean(r['f1_test'] for r in cok):.3f}"; macros["AffColdFailed"] = str(sum(1 for r in cold if r.get("failed")))

with open(os.path.join(OUT, "numbers.tex"), "w") as f:
    f.write("% generated by make_numbers.py -- do not edit\n")
    for k, v in sorted(macros.items()):
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
for k, v in sorted(macros.items()):
    print(f"{k:28s} {v}")
