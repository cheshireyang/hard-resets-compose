"""Wait until GPU is genuinely idle (no concurrent training like elastic) and Allen 78 cells exist,
then run the clean wall-clock GPU timing suite:
  1) E37 --mode timing (real Allen 69 firing cells: serial vs scan r0/r1/r2/bit-exact, fwd & full e2e iter)
  2) E11c (out-of-sample auto dispatch with recalibrated backward threshold T >= 4096)
  3) E32b (fair FFT-based batched serial baseline on real Allen cells)
"""
import glob, json, os, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable


def gpu_state():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
        text=True,
    ).strip().splitlines()[0]
    util, mem = [int(x.strip()) for x in out.split(",")]
    return util, mem


def elastic_running():
    out = subprocess.check_output(["ps", "aux"], text=True)
    return ("run_revision_cloud.py" in out) or ("train_imnet20.py" in out)


def allen78_ready():
    ids = set(json.load(open(os.path.join(ROOT, "results", "allen_cells_78.json"))))
    have = {int(os.path.basename(p)[:-4]) for p in glob.glob(os.path.join(ROOT, "data", "allen", "*.npz"))}
    return len(ids & have), len(ids)


def run(cmd, env_extra=None):
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    print(f"=== RUNNING: {' '.join(cmd)} ===", flush=True)
    t0 = time.time()
    rc = subprocess.call(cmd, cwd=ROOT, env=env)
    print(f"=== FINISHED rc={rc} in {time.time()-t0:.1f}s ===", flush=True)
    return rc


def main():
    idle_streak = 0
    while True:
        util, mem = gpu_state()
        el = elastic_running()
        n_have, n_tot = allen78_ready()
        print(f"[{time.strftime('%H:%M:%S')}] gpu_util={util}% mem={mem}MiB elastic={el} allen78={n_have}/{n_tot}", flush=True)
        if (not el) and util <= 5 and mem <= 2000 and n_have >= 69:
            idle_streak += 1
            if idle_streak >= 2:
                break
        else:
            idle_streak = 0
        time.sleep(60)

    print("GPU confirmed idle and Allen data ready! Starting clean wall-clock timing suite...", flush=True)
    run([PY, "-u", "experiments/e37_real_exact_e2e.py", "--mode", "timing"])
    run([PY, "-u", "experiments/e11c_auto_recal.py"])
    run([PY, "-u", "experiments/e32b_fair_batched.py"], env_extra={"E32_DATA": "data/allen"})
    print("ALL IDLE-GPU BENCHMARKS COMPLETE!", flush=True)


if __name__ == "__main__":
    main()
