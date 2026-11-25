"""E38: Same-cell, equal-length active-stimulus vs. silent-pause paired control on real Allen recordings.

Tests the causal mechanism behind E37's 134-round bit-exact repair cost:
Allen Noise-1 sweeps consist of three 3.07 s (60-chunk, C=256) active pink-noise epochs separated
by 4.92-8.96 s zero-current pauses (with a 6.96 s = 136-chunk trailing pause in 66/69 cells).

For each of the 69 valid firing Allen cells (with its fixed fitted hard-reset parameters), we run
the serial prefix on the full Noise-1 sweep to obtain exact entry membrane potentials, then compare
paired equal-length 60-chunk (L = 15,360 steps = 3.072 s, C = 256) windows from the exact same cell:
  - Active window: chunks [195, 255) (middle 3.07 s pink-noise stimulus epoch)
  - Silent window: chunks [255, 315) (immediately following 3.07 s zero-current pause)
(We also record the other two active/silent 60-chunk pairs [39,99) vs [99,159) and [351,411) vs
[411,471) as a robustness check, and report both C=256 matching E37 and C=128.)

On each window we measure:
  - firing rate, chunk-map pieces, overflow fraction
  - spike flips and max |u_scan - u_serial| at repair=0
  - exact repair rounds required for bit-identical U & S (repair=M)
  - idle-GPU pure forward wall-clock time (CUDA events): serial, scan(r0), scan(exact)

Output: results/e38_active_vs_silent.json
"""
import glob, json, math, os, subprocess, sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifscan import triton_lif as L
import experiments.e37_real_exact_e2e as E37

dev = "cuda"
C = 256
WIN_CHUNKS = 60          # 60 * 256 = 15,360 steps = 3.072 s
WIN_LEN = WIN_CHUNKS * C

PAIRS = (
    ("mid", 195, 255),   # primary pair: middle active [195,255) vs immediately following silent [255,315)
    ("early", 39, 99),   # robustness pair 1: first active [39,99) vs first silent [99,159)
    ("late", 351, 411),  # robustness pair 2: last active [351,411) vs trailing silent [411,471)
)


def check_gpu_idle():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
        text=True,
    ).strip().splitlines()[0]
    util, mem = [int(x.strip()) for x in out.split(",")]
    if util > 5 or mem > 2000:
        raise RuntimeError(f"GPU not idle: util={util}%, mem={mem} MiB")


def time_cuda(fn, reps=25):
    fn(); torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return float(np.median(ts))


def eval_window(I_win, u0_win, b, ur, U_ref, S_ref, C_use=C, do_time=True):
    T_w, N = I_win.shape
    M = T_w // C_use
    U_ser, S_ser = L.serial_forward_auto(I_win, b, 1.0, ur, u0_win)
    assert torch.equal(U_ser, U_ref) and torch.equal(S_ser, S_ref), "window serial != full prefix slice"

    st0 = {}
    U0, S0 = L.scan_forward_v3(I_win, b, 1.0, ur, u0_win, C=C_use, K=16, BN=32, nw=1, BNr=64, D=16, nwr=1, repair=0, stats=st0)
    stx = {}
    Ux, Sx = L.scan_forward_v3(I_win, b, 1.0, ur, u0_win, C=C_use, K=16, BN=32, nw=1, BNr=64, D=16, nwr=1, repair=M, stats=stx)

    chunk_spk = (S_ser.view(M, C_use, N).sum(1) > 0).cpu().numpy()
    max_sil = E37.max_silent_chunk_run(chunk_spk)

    out = dict(
        T=T_w, M=M, C=C_use,
        rate_hz=float(S_ser.mean().item() * 1000.0 / E37.DT_MS),
        n_spikes=int(S_ser.sum().item()),
        max_silent_chunks=int(max_sil),
        mean_pieces=float(st0["mean_pieces"]),
        overflow_frac=float(st0["overflow_frac"]),
        r0_spike_flips=int((S0 != S_ser).sum().item()),
        r0_bit_exact_spikes=bool(torch.equal(S0, S_ser)),
        r0_bit_exact_u=bool(torch.equal(U0, U_ser)),
        r0_max_u_err=float((U0 - U_ser).abs().max().item()),
        exact_repair_iters=int(stx["repair_iters"]),
        exact_bit_exact_u=bool(torch.equal(Ux, U_ser) and torch.equal(Sx, S_ser)),
    )
    if do_time:
        t_ser = time_cuda(lambda: L.serial_forward_auto(I_win, b, 1.0, ur, u0_win))
        t_r0 = time_cuda(lambda: L.scan_forward_v3(I_win, b, 1.0, ur, u0_win, C=C_use, K=16, BN=32, nw=1, BNr=64, D=16, nwr=1, repair=0))
        t_ex = time_cuda(lambda: L.scan_forward_v3(I_win, b, 1.0, ur, u0_win, C=C_use, K=16, BN=32, nw=1, BNr=64, D=16, nwr=1, repair=M))
        out["fwd_ms"] = dict(serial=t_ser, r0=t_r0, exact=t_ex)
        out["x_r0"] = float(t_ser / t_r0)
        out["x_exact"] = float(t_ser / t_ex)
    return out


def main():
    check_gpu_idle()
    e37a = json.load(open("results/e37_real_exact_acc.json"))["rows"]
    fits = {}
    for pat in ("results/e15all_allen_w*.json", "results/e15b_allen_w*.json", "results/e15c_allen_w*.json"):
        for f in sorted(glob.glob(pat)):
            for r in json.load(open(f)):
                m = r.get("models", {}).get("hard")
                if m and math.isfinite(m["beta"]) and 0 < m["beta"] < 1 and math.isfinite(m["gain"]) and m["gain"] > 0:
                    fits[r["cell"]["specimen"]] = m

    rows = []
    for r37 in e37a:
        sid = r37["specimen"]
        meta, Xtr, Str = E37.load_noise1_with_spikes(f"data/allen/{sid}.npz")
        p = fits[sid]
        b, gain, bias, ur = p["beta"], p["gain"], p["bias"], p["ur"]
        I = (gain * Xtr + bias).contiguous()
        T, N = I.shape
        u0 = torch.zeros(N, device=dev)
        U_full, S_full = L.serial_forward_auto(I, b, 1.0, ur, u0)

        rec = dict(specimen=sid, dendrite=meta.get("dendrite", ""), T=T, N=N, beta=b, ur=ur, pairs={})
        for pname, a_ch, s_ch in PAIRS:
            t_a0, t_a1 = a_ch * C, (a_ch + WIN_CHUNKS) * C
            t_s0, t_s1 = s_ch * C, (s_ch + WIN_CHUNKS) * C
            do_t = pname in ("mid", "late")
            act = eval_window(I[t_a0:t_a1].contiguous(), U_full[t_a0 - 1].contiguous(), b, ur,
                              U_full[t_a0:t_a1], S_full[t_a0:t_a1], C_use=C, do_time=do_t)
            sil = eval_window(I[t_s0:t_s1].contiguous(), U_full[t_s0 - 1].contiguous(), b, ur,
                              U_full[t_s0:t_s1], S_full[t_s0:t_s1], C_use=C, do_time=do_t)
            rec["pairs"][pname] = dict(active=act, silent=sil)
        rows.append(rec)
        ma, ms = rec["pairs"]["late"]["active"], rec["pairs"]["late"]["silent"]
        print(
            f"cell {sid} N={N} | LATE ACT: rate={ma['rate_hz']:5.1f}Hz flips={ma['r0_spike_flips']} rep={ma['exact_repair_iters']:2d} "
            f"fwd ser={ma['fwd_ms']['serial']:.2f} r0={ma['fwd_ms']['r0']:.2f} ex={ma['fwd_ms']['exact']:.2f} ({ma['x_exact']:.2f}x) | "
            f"SIL: rep={ms['exact_repair_iters']:2d} ex={ms['fwd_ms']['exact']:.2f} ({ms['x_exact']:.2f}x)",
            flush=True,
        )

    # Summary statistics + paired bootstrap 95% CIs on primary pair ("mid")
    rng = np.random.default_rng(0)
    def boot_med(vals):
        arr = np.asarray(vals, dtype=np.float64)
        idx = rng.integers(0, len(arr), size=(10000, len(arr)))
        ms = np.median(arr[idx], axis=1)
        return dict(med=float(np.median(arr)), lo=float(np.quantile(ms, 0.025)), hi=float(np.quantile(ms, 0.975)),
                    min=float(arr.min()), max=float(arr.max()))

    summary = dict(cells=len(rows), C=C, win_chunks=WIN_CHUNKS, win_steps=WIN_LEN, pairs={})
    for pname, _, _ in PAIRS:
        acts = [r["pairs"][pname]["active"] for r in rows]
        sils = [r["pairs"][pname]["silent"] for r in rows]
        ps = dict(
            act_flips_zero=sum(a["r0_bit_exact_spikes"] for a in acts),
            sil_flips_zero=sum(s["r0_bit_exact_spikes"] for s in sils),
            act_exact_u_ok=sum(a["exact_bit_exact_u"] for a in acts),
            sil_exact_u_ok=sum(s["exact_bit_exact_u"] for s in sils),
            act_repair_iters=boot_med([a["exact_repair_iters"] for a in acts]),
            sil_repair_iters=boot_med([s["exact_repair_iters"] for s in sils]),
            repair_diff_sil_minus_act=boot_med([s["exact_repair_iters"] - a["exact_repair_iters"] for a, s in zip(acts, sils)]),
            sil_gt_act_cells=sum(s["exact_repair_iters"] > a["exact_repair_iters"] for a, s in zip(acts, sils)),
            act_rate_hz_med=float(np.median([a["rate_hz"] for a in acts])),
            sil_rate_hz_med=float(np.median([s["rate_hz"] for s in sils])),
            act_pieces_mean=float(np.mean([a["mean_pieces"] for a in acts])),
            sil_pieces_mean=float(np.mean([s["mean_pieces"] for s in sils])),
            act_ovf_mean=float(np.mean([a["overflow_frac"] for a in acts])),
            sil_ovf_mean=float(np.mean([s["overflow_frac"] for s in sils])),
        )
        if pname in ("mid", "late"):
            ps["act_fwd_ser_ms"] = boot_med([a["fwd_ms"]["serial"] for a in acts])
            ps["act_fwd_r0_ms"] = boot_med([a["fwd_ms"]["r0"] for a in acts])
            ps["act_fwd_exact_ms"] = boot_med([a["fwd_ms"]["exact"] for a in acts])
            ps["sil_fwd_ser_ms"] = boot_med([s["fwd_ms"]["serial"] for s in sils])
            ps["sil_fwd_r0_ms"] = boot_med([s["fwd_ms"]["r0"] for s in sils])
            ps["sil_fwd_exact_ms"] = boot_med([s["fwd_ms"]["exact"] for s in sils])
            ps["act_x_r0"] = boot_med([a["x_r0"] for a in acts])
            ps["act_x_exact"] = boot_med([a["x_exact"] for a in acts])
            ps["sil_x_r0"] = boot_med([s["x_r0"] for s in sils])
            ps["sil_x_exact"] = boot_med([s["x_exact"] for s in sils])
        summary["pairs"][pname] = ps

    print("SUMMARY LATE:", json.dumps(summary["pairs"]["late"], indent=2), flush=True)
    out_path = os.path.join(ROOT, "results", "e38_active_vs_silent.json")
    json.dump(dict(summary=summary, rows=rows), open(out_path, "w"), indent=1)
    print("Saved", out_path, flush=True)


if __name__ == "__main__":
    main()
