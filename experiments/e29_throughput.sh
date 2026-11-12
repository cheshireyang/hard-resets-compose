#!/bin/bash
# E29: database-scale throughput, ALIF fits of the first 64 cells (E28 protocol), serial kernels vs scans, with
# NW concurrent processes under CUDA MPS.  Wall-clock per configuration -> results/e29_throughput.json
cd "$(dirname "$0")/.."
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log
echo "[" > results/e29_throughput.json; first=1
for cfg in "scan 8" "scan 16" "serial 8" "serial 16" "serial 32"; do
  set -- $cfg; MODE=$1; NW=$2
  t0=$(date +%s%3N)
  for W in $(seq 0 $((NW-1))); do
    E28_BASE_GLOB="results/e15all_allen_w*.json" E28_GLIF=results/allen_glif_models_all.json E28_DATA=data/allen_all E28_TAG=tp_${MODE}_${NW} \
      E28_TIMING=0 E28_MODE=$MODE E28_LIMIT=64 python3 experiments/e28_allen_alif.py $W $NW > logs/e29_${MODE}_${NW}_w$W.log 2>&1 &
  done
  wait
  t1=$(date +%s%3N)
  [ $first = 1 ] || echo "," >> results/e29_throughput.json; first=0
  echo "{\"mode\": \"$MODE\", \"workers\": $NW, \"cells\": 64, \"ms\": $((t1 - t0))}" >> results/e29_throughput.json
  echo "$MODE $NW: $((t1 - t0)) ms"
done
echo "]" >> results/e29_throughput.json
