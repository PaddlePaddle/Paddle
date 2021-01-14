#!/bin/bash
set -e
# use default values
# FIXME: random fails on Unknown command lines -c (or -m).
launch_py=${PADDLE_BINARY_DIR}/python/paddle/distributed/launch.py
CUDA_VISIBLE_DEVICES=0,1 python ${launch_py} c_comm_init_op.py
