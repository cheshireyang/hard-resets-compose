cd "$(dirname "$0")/.."
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log
NW=8
for W in $(seq 0 $((NW-1))); do
  E15_KINDS=hard,soft E15_DATA="data/allen_all/*.npz" E15_TIMING=0 E15_OUT=results/e15all_allen_w$W.json python3 experiments/e15_allen_fit.py 0.2 600 3 $W $NW > logs/e15all_w$W.log 2>&1 &
done
wait
for W in $(seq 0 $((NW-1))); do
  E28_BASE_GLOB="results/e15all_allen_w*.json" E28_GLIF=results/allen_glif_models_all.json E28_DATA=data/allen_all E28_TAG=all E28_TIMING=0 \
    python3 experiments/e28_allen_alif.py $W $NW > logs/e28all_w$W.log 2>&1 &
done
wait
until [ -f logs/e27all.done ]; do sleep 20; done
E28_BASE_GLOB="results/e15all_allen_w*.json" E28_GLOB="results/e28all_allen_alif_w*.json" E27_DATA=data/allen_all E27_OUT=results/e27all_glif_official.json \
  E27_SPK=results/e27all_spikes E27B_OUT=results/e27ball_metrics.json python3 experiments/e27b_metrics.py > logs/e27ball.log 2>&1 && echo DONE > logs/q_all.done
