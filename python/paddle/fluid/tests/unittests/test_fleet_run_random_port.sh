#!/bin/bash
set -e

#test for random ports
file_0_0="test_launch_filelock_0_0.log"
file_1_0="test_launch_filelock_1_0.log"
rm -rf $file_0_0 $file_0_1

distributed_args="--gpus=0,1 --log_dir=testlog"
export PADDLE_LAUNCH_LOG="test_launch_filelock_0"
CUDA_VISIBLE_DEVICES=0,1 fleetrun ${distributed_args} find_ports.py
str_0="worker_endpoints:127.0.0.1:6070,127.0.0.1:6071"