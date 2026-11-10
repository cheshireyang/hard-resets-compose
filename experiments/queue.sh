#!/bin/sh
# GPU job queue (run on the CUDA box from the project root). One job at a time -> clean timings.
cd "$(dirname "$0")/.."
mkdir -p results logs
rm -f results/e6b_fair_beta0.9.json results/e6b_fair_beta0.99.json
python3 experiments/e6b_gpu_fair.py 0.9  > logs/e6b_0.9.log 2>&1
python3 experiments/e6b_gpu_fair.py 0.99 > logs/e6b_0.99.log 2>&1
for s in 0 1 2; do for m in serial scan sj; do
  python3 experiments/e7_shd.py $m 1024 $s 30 64 > logs/e7_${m}_T1024_s$s.log 2>&1
done; done
python3 experiments/e7b_gradcheck.py 8192 > logs/e7b_8192.log 2>&1
for m in serial scan sj; do
  python3 experiments/e7_shd.py $m 8192 0 20 16 > logs/e7_${m}_T8192_s0.log 2>&1
done
echo QUEUE_DONE > logs/queue.done
