#!/bin/sh
cd "$(dirname "$0")/.."
while [ ! -f logs/e17.done ]; do sleep 20; done
python3 experiments/e18_soft_allen.py > logs/e18.log 2>&1
echo DONE > logs/e18.done
