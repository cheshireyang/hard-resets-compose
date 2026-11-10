#!/bin/sh
# balanced, larger Allen experiment: fetch, then 6 parallel fitting workers (no timing; timing is from e15 v1)
cd "$(dirname "$0")/.."
python3 experiments/allen_fetch_balanced.py 30 > logs/allen_fetch_bal.log 2>&1
for w in 0 1 2 3 4 5; do
  E15_TIMING=0 E15_OUT=results/e15b_allen_w$w.json python3 experiments/e15_allen_fit.py 0.2 400 3 $w 6 > logs/e15b_w$w.log 2>&1 &
done
wait
echo DONE > logs/allen2.done
