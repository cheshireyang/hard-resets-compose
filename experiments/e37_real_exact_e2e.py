"""E37: Unified exactness + end-to-end latency audit on real Allen Cell Types recordings.

On the same batch of real mouse visual cortex Noise-1 current-clamp recordings (the 78-cell Allen
cohort of E15t/E15b/c, filtered to cells with valid finite fitted parameters; plus rate-matched
training-init evaluation on all 78 cells), measures side-by-side for:
  - tuned serial kernel (`serial`)
  - exact map scan, no repair (`scan_r0`)
  - exact map scan, 1 repair round (`scan_r1`)
  - exact map scan, 2 repair rounds (`scan_r2`)
  - exact map scan, uncapped bit-exact repair (`scan_exact`, repair=M until U and S are bit-identical)

Accuracy metrics (100% deterministic, unaffected by GPU load):
  - spike mismatch fraction & count vs serial
  - max |u_scan - u_serial| and 99.9th-percentile |u_scan - u_serial|
  - bit-exactness of S and of full membrane potential trajectory U
  - repair rounds needed for bit-exact U, alongside max consecutive silent-chunk run
  - chunk map stats (mean pieces, max pieces, overflow fraction)
  - end-to-end parameter gradient relative error (beta, gain, bias, u_r) through van Rossum FFT loss

Timing metrics (--mode timing; requires idle GPU unless --allow-busy is passed):
  - pure forward latency (ms) for serial, scan_r0, scan_r1, scan_exact
  - full end-to-end training iteration latency (forward + van Rossum FFT loss + rate term +
    backward + Adam step; median of iters 5..29 of a 30-iter fit) for serial, scan_r0, scan_r1,
    scan_r2, and scan_exact.

Usage:
  python3 experiments/e37_real_exact_e2e.py --mode accuracy [--data data/allen] [--out results/e37_real_exact_acc.json]
  python3 experiments/e37_real_exact_e2e.py --mode timing   [--data data/allen] [--out results/e37_real_exact_timing.json]
"""
import argparse, glob, json, math, os, subprocess, sys, time
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifscan import triton_lif as L
from lifscan.fit import lif_param

DT_MS = 0.2
TAU_VR_MS = 10.0
dev = "cuda"


def load_noise1_with_spikes(path):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    sweeps = sorted({k.split("__")[0] for k in z.files if "__" in k})
    pairs = []
    for s in sweeps:
        if str(z[f"{s}__name"]) != "Noise 1":
            continue
        rate = float(z[f"{s}__rate"])
        i_pa = z[f"{s}__i_pa"]
        sp = z[f"{s}__spikes"]
        step = int(round(rate * DT_MS / 1000.0))
        n = len(i_pa) // step
        x = i_pa[: n * step].reshape(n, step).mean(1)
        s_bin = np.zeros(n, dtype=np.float32)
        idx = np.minimum((sp * 1000.0 / DT_MS).astype(np.int64), n - 1)
        s_bin[idx] = 1.0
        pairs.append((x, s_bin))
    if len(pairs) < 2:
        return None
    T = (min(len(p[0]) for p in pairs) // 256) * 256
    X = np.stack([p[0][:T] for p in pairs], 1).astype(np.float32) / 100.0
    S = np.stack([p[1][:T] for p in pairs], 1).astype(np.float32)
    return meta, torch.tensor(X, device=dev), torch.tensor(S, device=dev)


def vr(S, tau_steps):
    a = math.exp(-1.0 / tau_steps)
    k = a ** torch.arange(S.shape[0], device=S.device, dtype=S.dtype)
    n = 2 * S.shape[0]
    return torch.fft.irfft(torch.fft.rfft(S, n=n, dim=0) * torch.fft.rfft(k, n=n)[:, None], n=n, dim=0)[: S.shape[0]]


def max_silent_chunk_run(S_bool_MN):
    """S_bool_MN: [M, N] bool array (True if chunk has >=1 spike). Returns max consecutive False run."""
    M, N = S_bool_MN.shape
    best = 0
    for j in range(N):
        cur = 0
        for m in range(M):
            if S_bool_MN[m, j]:
                cur = 0
            else:
                cur += 1
                if cur > best:
                    best = cur
    return int(best)


def rate_matched_init(Xtr, target, seed=7):
    g = torch.Generator().manual_seed(seed)
    beta0 = 0.95 + 0.045 * torch.rand(1, generator=g).item()
    lo, hi = math.log(1e-4), math.log(1.0)
    u0 = torch.zeros(Xtr.shape[1], device=dev)
    for _ in range(22):
        mid = 0.5 * (lo + hi)
        I = (math.exp(mid) * Xtr).contiguous()
        with torch.no_grad():
            _, S = L.serial_forward_auto(I, beta0, 1.0, 0.0, u0)
            r = S.mean().item()
        if r > target:
            hi = mid
        else:
            lo = mid
    return dict(beta=beta0, gain=math.exp(0.5 * (lo + hi)), bias=0.0, ur=0.0)


def grad_vector(Xtr, Ytr, target, params_dict, mode, repair=0):
    b = params_dict["beta"]
    zb = torch.tensor(math.log(b / (1.0 - b)), device=dev, requires_grad=True)
    lg = torch.tensor(math.log(max(params_dict["gain"], 1e-8)), device=dev, requires_grad=True)
    bias = torch.tensor(params_dict["bias"], device=dev, requires_grad=True)
    ur = torch.tensor(params_dict["ur"], device=dev, requires_grad=True)
    u0 = torch.zeros(Xtr.shape[1], device=dev)
    I = torch.exp(lg) * Xtr + bias
    S = lif_param(I, u0, torch.sigmoid(zb), torch.tensor(1.0, device=dev), ur, mode=mode, K=16, repair=repair)
    tau = TAU_VR_MS / DT_MS
    loss = ((vr(S, tau) - Ytr) ** 2).mean() / (Ytr ** 2).mean().clamp(min=1e-12) + ((S.mean() - target) / max(target, 1e-12)) ** 2
    loss.backward()
    return torch.stack([zb.grad.detach(), lg.grad.detach(), bias.grad.detach(), ur.grad.detach()]), float(loss.item())


def eval_cell_accuracy(meta, Xtr, Str, params_dict, fit_valid):
    T, N = Xtr.shape
    C = L._pick_C(T, D=16)
    M = T // C
    u0 = torch.zeros(N, device=dev)
    b, gain, bias, ur = params_dict["beta"], params_dict["gain"], params_dict["bias"], params_dict["ur"]
    I = (gain * Xtr + bias).contiguous()

    U_ser, S_ser = L.serial_forward_auto(I, b, 1.0, ur, u0)
    rate = float(S_ser.mean().item())
    chunk_has_spike = (S_ser.view(M, C, N).sum(dim=1) > 0).cpu().numpy()
    max_silent = max_silent_chunk_run(chunk_has_spike)
    frac_silent = float((~chunk_has_spike).mean())

    tau = TAU_VR_MS / DT_MS
    Ytr = vr(Str, tau)
    target = float(Str.mean().item())
    g_ser, loss_ser = grad_vector(Xtr, Ytr, target, params_dict, mode="serial")
    g_ser_norm = float(g_ser.norm().item())

    modes = {}
    for label, rep in (("r0", 0), ("r1", 1), ("r2", 2), ("exact", M)):
        st = {}
        U_sc, S_sc = L.scan_forward_v3(
            I, b, 1.0, ur, u0, C=C, K=16, BN=32, nw=1, BNr=64, D=16, nwr=1, repair=rep, stats=st
        )
        diff_u = (U_sc - U_ser).abs()
        max_u = float(diff_u.max().item())
        # subsample if huge for quantile safety, though T*N ~ 5e5 is tiny
        p999_u = float(torch.quantile(diff_u.flatten(), 0.999).item())
        flip_cnt = int((S_sc != S_ser).sum().item())
        flip_frac = float(flip_cnt / S_ser.numel())
        g_sc, loss_sc = grad_vector(Xtr, Ytr, target, params_dict, mode="scan", repair=rep)
        g_rel = float((g_sc - g_ser).norm().item() / max(g_ser_norm, 1e-12))
        modes[label] = dict(
            repair_cap=rep,
            repair_iters=int(st["repair_iters"]),
            mean_pieces=float(st["mean_pieces"]),
            max_pieces=int(st["max_pieces"]),
            overflow_frac=float(st["overflow_frac"]),
            spike_flips=flip_cnt,
            spike_mismatch_frac=flip_frac,
            bit_exact_spikes=bool(torch.equal(S_sc, S_ser)),
            bit_exact_u=bool(torch.equal(U_sc, U_ser)),
            max_u_err=max_u,
            p999_u_err=p999_u,
            grad_rel_err=g_rel,
            loss_abs_err=abs(loss_sc - loss_ser),
        )
    return dict(
        specimen=meta["specimen"], dendrite=meta.get("dendrite", ""),
        fit_valid=fit_valid, T=T, N=N, C=C, M=M,
        beta=b, ur=ur, model_rate=rate,
        max_silent_chunks=max_silent, frac_silent_chunks=frac_silent,
        modes=modes,
    )


def time_fit_loop(Xtr, Str, mode, repair=0, iters=30, seed=7):
    target = float(Str.mean().item())
    init = rate_matched_init(Xtr, target, seed=seed)
    p = dict(
        zb=torch.tensor(math.log(init["beta"] / (1.0 - init["beta"])), device=dev, requires_grad=True),
        lg=torch.tensor(math.log(init["gain"]), device=dev, requires_grad=True),
        bias=torch.tensor(0.0, device=dev, requires_grad=True),
        ur=torch.tensor(0.0, device=dev, requires_grad=True),
    )
    opt = torch.optim.Adam([p["zb"], p["lg"], p["bias"], p["ur"]], lr=0.005)
    tau = TAU_VR_MS / DT_MS
    Ytr = vr(Str, tau)
    u0 = torch.zeros(Xtr.shape[1], device=dev)
    times = []
    for _ in range(iters):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        I = torch.exp(p["lg"]) * Xtr + p["bias"]
        S = lif_param(I, u0, torch.sigmoid(p["zb"]), torch.tensor(1.0, device=dev), p["ur"],
                      mode=mode, K=16, repair=repair)
        loss = ((vr(S, tau) - Ytr) ** 2).mean() / (Ytr ** 2).mean() + ((S.mean() - target) / target) ** 2
        opt.zero_grad(); loss.backward(); opt.step()
        torch.cuda.synchronize(); times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times[5:]))


def time_fwd_only(I, b, ur, mode, repair=0, reps=15):
    T, N = I.shape
    u0 = torch.zeros(N, device=dev)
    C = L._pick_C(T, D=16)
    def run():
        if mode == "serial":
            return L.serial_forward_auto(I, b, 1.0, ur, u0)
        return L.scan_forward_v3(I, b, 1.0, ur, u0, C=C, K=16, BN=32, nw=1, BNr=64, D=16, nwr=1, repair=repair)
    run(); torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); run(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return float(np.median(ts))


def check_gpu_idle():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
        text=True,
    ).strip().splitlines()[0]
    util, mem = [int(x.strip()) for x in out.split(",")]
    return util, mem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("accuracy", "timing"), required=True)
    ap.add_argument("--data", default="data/allen")
    ap.add_argument("--ids", default="results/allen_cells_78.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--allow-busy", action="store_true")
    args = ap.parse_args()

    if args.mode == "timing" and not args.allow_busy:
        util, mem = check_gpu_idle()
        if util > 5 or mem > 2000:
            raise RuntimeError(f"GPU is NOT idle (util={util}%, mem={mem} MiB). Aborting timing run!")

    out_path = args.out or (
        "results/e37_real_exact_acc.json" if args.mode == "accuracy" else "results/e37_real_exact_timing.json"
    )
    target_ids = set(json.load(open(args.ids))) if args.ids and os.path.exists(args.ids) else None

    # Load fitted hard-reset parameters (prefer e15b/c for the 78-cell cohort, fallback to e15all)
    fits = {}
    for pat in ("results/e15all_allen_w*.json", "results/e15b_allen_w*.json", "results/e15c_allen_w*.json"):
        for f in sorted(glob.glob(pat)):
            for r in json.load(open(f)):
                m = r.get("models", {}).get("hard")
                if m and math.isfinite(m["beta"]) and 0.0 < m["beta"] < 1.0 and math.isfinite(m["gain"]) and m["gain"] > 0:
                    fits[r["cell"]["specimen"]] = m

    paths = sorted(glob.glob(f"{args.data}/*.npz"))
    rows = []
    for path in paths:
        sid = int(os.path.basename(path)[:-4])
        if target_ids is not None and sid not in target_ids:
            continue
        loaded = load_noise1_with_spikes(path)
        if loaded is None:
            continue
        meta, Xtr, Str = loaded
        if float(Str.sum().item()) == 0.0:
            print(f"cell {sid}: skipping (0 spikes in Noise 1)", flush=True)
            continue
        if sid in fits:
            pdict, fit_valid = fits[sid], True
        else:
            pdict, fit_valid = rate_matched_init(Xtr, float(Str.mean().item()), seed=7), False

        if args.mode == "accuracy":
            row = eval_cell_accuracy(meta, Xtr, Str, pdict, fit_valid)
            r0, rx = row["modes"]["r0"], row["modes"]["exact"]
            print(
                f"cell {sid} ({'fit' if fit_valid else 'ini'}) T={row['T']} N={row['N']} silent={row['max_silent_chunks']:3d} | "
                f"r0: flips={r0['spike_flips']} maxU={r0['max_u_err']:.2e} grad={r0['grad_rel_err']:.2e} ovf={100*r0['overflow_frac']:.1f}% | "
                f"exact: iters={rx['repair_iters']:3d} eqU={rx['bit_exact_u']} grad={rx['grad_rel_err']:.2e}",
                flush=True,
            )
            rows.append(row)
        else:
            T, N = Xtr.shape
            C = L._pick_C(T, D=16)
            M = T // C
            I = (pdict["gain"] * Xtr + pdict["bias"]).contiguous()
            b, ur = pdict["beta"], pdict["ur"]
            fwd_ser = time_fwd_only(I, b, ur, "serial")
            fwd_r0 = time_fwd_only(I, b, ur, "scan", repair=0)
            fwd_r1 = time_fwd_only(I, b, ur, "scan", repair=1)
            fwd_r2 = time_fwd_only(I, b, ur, "scan", repair=2)
            fwd_ex = time_fwd_only(I, b, ur, "scan", repair=M)
            e2e_ser = time_fit_loop(Xtr, Str, "serial", repair=0)
            e2e_r0 = time_fit_loop(Xtr, Str, "scan", repair=0)
            e2e_r1 = time_fit_loop(Xtr, Str, "scan", repair=1)
            e2e_r2 = time_fit_loop(Xtr, Str, "scan", repair=2)
            e2e_ex = time_fit_loop(Xtr, Str, "scan", repair=M)
            row = dict(
                specimen=sid, dendrite=meta.get("dendrite", ""), fit_valid=fit_valid,
                T=T, N=N, C=C, M=M,
                fwd_ms=dict(serial=fwd_ser, r0=fwd_r0, r1=fwd_r1, r2=fwd_r2, exact=fwd_ex),
                e2e_ms=dict(serial=e2e_ser, r0=e2e_r0, r1=e2e_r1, r2=e2e_r2, exact=e2e_ex),
            )
            print(
                f"cell {sid} T={T} N={N} | e2e ms: ser={e2e_ser:.2f} r0={e2e_r0:.2f} ({e2e_ser/e2e_r0:.2f}x) "
                f"r1={e2e_r1:.2f} ({e2e_ser/e2e_r1:.2f}x) exact={e2e_ex:.2f} ({e2e_ser/e2e_ex:.2f}x)",
                flush=True,
            )
            rows.append(row)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        json.dump(dict(mode=args.mode, cells=len(rows), rows=rows), open(out_path, "w"), indent=1)

    print(f"Saved {len(rows)} cells to {out_path}", flush=True)


if __name__ == "__main__":
    main()
