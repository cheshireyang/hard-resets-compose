#!/bin/sh
cd "$(dirname "$0")/.."
while [ ! -f logs/queue.done ]; do sleep 30; done
python3 experiments/e6b_gpu_fair.py 0.9  > logs/e6c_0.9.log 2>&1
python3 experiments/e6b_gpu_fair.py 0.99 > logs/e6c_0.99.log 2>&1
python3 experiments/e7c_timing.py > logs/e7c.log 2>&1
echo DONE > logs/queue2.done
