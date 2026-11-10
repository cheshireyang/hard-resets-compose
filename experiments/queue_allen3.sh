#!/bin/sh
# 16 parallel fitting workers; cells already fitted by the e15b run are skipped
cd "$(dirname "$0")/.."
for w in 0 1; do
  E15_TIMING=0 E15_SKIP_GLOB="results/e15b_allen_w*.json" E15_OUT=results/e15c_allen_w$w.json \
    python3 experiments/e15_allen_fit.py 0.2 400 3 $w 2 > logs/e15c_w$w.log 2>&1 &
done
wait
echo DONE > logs/allen3.done
