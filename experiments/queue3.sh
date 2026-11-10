#!/bin/sh
# extra seeds for the SHD accuracy comparison (runs after queue2)
cd "$(dirname "$0")/.."
while [ ! -f logs/queue2.done ]; do sleep 30; done
for s in 3 4 5 6 7; do for m in serial scan sj; do
  python3 experiments/e7_shd.py $m 1024 $s 30 64 > logs/e7_${m}_T1024_s$s.log 2>&1
done; done
echo DONE > logs/queue3.done
