#!/bin/bash
set -e
# use default values
# FIXME: random fails on Unknown command lines -c (or -m).
launch_py=${PADDLE_BINARY_DIR}/python/paddle/distributed/launch.py
python ${launch_py} paddle_distributed.py "nccl"
python ${launch_py} paddle_distributed.py "gloo"
