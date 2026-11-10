#!/bin/sh
cd "$(dirname "$0")/.."
python3 experiments/allen_fetch.py 20 > logs/allen_fetch20.log 2>&1
python3 experiments/e15_allen_fit.py 0.2 400 3 > logs/e15_allen.log 2>&1
echo DONE > logs/allen.done
