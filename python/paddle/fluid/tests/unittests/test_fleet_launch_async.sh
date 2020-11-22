#!/bin/bash
set -e

# test use DISTRIBUTED_TRAINER_ENDPOINTS env in paddlecloud
unset PADDLE_PORT
export DISTRIBUTED_TRAINER_ENDPOINTS=127.0.0.1:6170,127.0.0.1:6171,127.0.0.2:6170,127.0.0.2:6171

echo "paddle.distributed.fleet.launch async poll process test"
if ! CUDA_VISIBLE_DEVICES=0,1 fleetrun ${distributed_args} multi_process.py fleetrun abort; then
    echo "train abort as planned"
fi

abort_str1="abort>>> selected_gpus:0 worker_endpoints:127.0.0.1:6170,127.0.0.1:6171,127.0.0.2:6170,127.0.0.2:6171 trainers_num:4 current_endpoint:127.0.0.1:6170 trainer_id:0"

if grep -q "$abort_str1" "$file_0"; then
    echo "trainer 0 abort as planned"
else
    echo "trainer 0 not abort as planned"
    exit -1
fi

if [ ! -f $file_1 ]; then
    echo "trainer 1 terminate as planned"
else
    echo "trainer 1 not terminate as planned"
    exit -1
fi