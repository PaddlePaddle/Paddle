#!/bin/bash
set -e
# use default values
# FIXME: random fails on Unknown command lines -c (or -m).
#launch_py=${PADDLE_BINARY_DIR}/python/paddle/distributed/launch.py
python -m paddle.distributed.launch --selected_gpus="0,1" c_comm_init_op.py
