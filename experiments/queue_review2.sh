#!/bin/bash
# Review-2 GPU queue (run on a fresh RTX 5090 pod from the project root):
#   tests -> E31 (long single recordings, incl. CPU baseline) -> E11b (dispatch, out of sample)
#   -> Allen data fetch -> E33b (downward-reset check, CPU) -> E32t (per-cell timing, same GPU) -> E32 (batched serial fits of the whole database)
# Each step writes a done marker only on success; rerunning skips finished steps.
cd "$(dirname "$0")/.."
mkdir -p logs results
step() {  # step NAME CMD...
  local name=$1; shift
  if [ -f logs/$name.done ]; then echo "skip $name"; return 0; fi
  echo "=== $name: $*"
  if "$@" > logs/$name.log 2>&1; then touch logs/$name.done; echo "ok $name"; else echo "FAILED $name (see logs/$name.log)"; return 1; fi
}
step test_triton python3 tests/test_triton.py || exit 1
grep -q "ALL OK" logs/test_triton.log || { echo "triton tests failed"; rm -f logs/test_triton.done; exit 1; }
step test_batch  python3 tests/test_batch.py || exit 1
grep -q "BATCH OK" logs/test_batch.log || { echo "batch kernels not exact"; rm -f logs/test_batch.done; exit 1; }
step e31_b09  python3 experiments/e31_long_single.py 0.9
step e31_b099 python3 experiments/e31_long_single.py 0.99
step e11b env E11_N=16,64,256,2048,4096,16384 E11_T=2048,8192,32768,131072 E11_OUT=results/e11b_auto_oos.json python3 experiments/e11_auto.py
step fetch env ALLEN_OUT=data/allen_all python3 experiments/allen_fetch_all.py 32
step e33b python3 experiments/e33b_allen_downward.py
step e32t env NCELLS=32 python3 experiments/e32t_percell_timing.py
step e32_hard env MODEL=hard python3 experiments/e32_batched.py
step e32_alif env MODEL=alif python3 experiments/e32_batched.py
step e36 python3 experiments/e36_real_inputs.py
echo "queue finished"
