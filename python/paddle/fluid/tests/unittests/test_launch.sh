#!/bin/bash
set -e

# use default values
python -m paddle.distributed.launch multi_process.py

# use specified values
cluster_node_ips="127.0.0.1"
node_ip="127.0.0.1"

distributed_args="--cluster_node_ips ${cluster_node_ips} --node_ip ${node_ip} --selected_gpus=0,1 --log_dir testlog"
python -m paddle.distributed.launch ${distributed_args} multi_process.py

str1="selected_gpus:0 worker_endpoints:['127.0.0.1:6170', '127.0.0.1:6171'] trainers_num:2 current_endpoint:127.0.0.1:6170 trainer_id:0"
str2="selected_gpus:1 worker_endpoints:['127.0.0.1:6170', '127.0.0.1:6171'] trainers_num:2 current_endpoint:127.0.0.1:6171 trainer_id:1"
file="multi_process.check.log"

if ! grep -q "$str1" "$file"; then
    echo "find trainer 0"
else
    echo "not find trainer 0"
    exit -1
fi

if ! grep -q "$str2" "$file"; then
    echo "find trainer 1"
else
    echo "not find trainer 0"
    exit -1
fi
