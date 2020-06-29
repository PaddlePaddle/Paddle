#!/bin/bash
set -e

# Test data norm op with sync_state set to True
launch_py=${PADDLE_BINARY_DIR}/python/paddle/distributed/launch.py
python ${launch_py} --selected_gpus="0,1" multi_process_data_norm.py
