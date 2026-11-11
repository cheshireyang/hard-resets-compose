#!/bin/sh
cd "$(dirname "$0")/.."
python3 experiments/allen_fetch.py 20 > logs/allen_fetch20b.log 2>&1
python3 experiments/e22_allen_affine.py 0 2 > logs/e22_w0.log 2>&1 &
python3 experiments/e22_allen_affine.py 1 2 > logs/e22_w1.log 2>&1 &
wait
echo DONE > logs/e22.done
