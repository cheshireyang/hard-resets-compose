#!/bin/sh
# E14 on GPU: long-recording fits, serial vs auto (one job at a time)
cd "$(dirname "$0")/.."
python3 experiments/e14_fit.py cuda 65536   16 300 2 hard:serial hard:auto > logs/e14_T65536.log 2>&1
python3 experiments/e14_fit.py cuda 262144  16 300 2 hard:serial hard:auto > logs/e14_T262144.log 2>&1
python3 experiments/e14_fit.py cuda 1048576 16 30  1 hard:serial hard:auto > logs/e14_T1048576.log 2>&1
echo DONE > logs/fit.done
