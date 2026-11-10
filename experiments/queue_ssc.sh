#!/bin/sh
# SSC experiments (run after the E9 benchmark so timings stay clean)
cd "$(dirname "$0")/.."
while [ ! -f logs/e9.done ]; do sleep 30; done
export DATASET=ssc
python3 experiments/e7b_gradcheck.py 1024 > logs/e10_gradcheck.log 2>&1
for s in 0 1 2; do for m in serial scan sj; do
  python3 experiments/e7_shd.py $m 1024 $s 20 64 > logs/e10_${m}_s$s.log 2>&1
done; done
echo DONE > logs/ssc.done
