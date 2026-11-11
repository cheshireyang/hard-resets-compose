#!/bin/sh
cd "$(dirname "$0")/.."
python3 experiments/e21_chunk_restart.py > logs/e21.log 2>&1
python3 experiments/e20_fpt_psn.py > logs/e20.log 2>&1
echo DONE > logs/h2h.done
