#!/bin/bash
export PYTHONUNBUFFERED=1 
export CUDA_VISIBLE_DEVICES=7 
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:/home/reyoung/cuda/lib64
export BATCH_SIZE=32
export PARALLEL=0

echo "Generating data and warming up. Run train.py without profiler"
DROP_PICKLE=1  /usr/bin/time -v python train.py 1>/dev/null # generate cache data
echo "Run train.py with profiler"
DROP_PICKLE=0 python -m yep train.py

echo "use pprof -http 0.0.0.0:[PORT] (which python) train.py.prof to show flamegraph"
