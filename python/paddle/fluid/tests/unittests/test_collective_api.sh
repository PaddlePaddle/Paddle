#!/bin/bash
set -e
# use default values
# FIXME: random fails on Unknown command lines -c (or -m).
launch_py=${PADDLE_BINARY_DIR}/python/paddle/distributed/launch.py
#CUDA_VISIBLE_DEVICE="0,1" python ${launch_py} --selected_gpus="0,1" paddle_distributed.py "nccl"
python ${launch_py} --selected_gpus="0,1" paddle_distributed.py "gloo"
