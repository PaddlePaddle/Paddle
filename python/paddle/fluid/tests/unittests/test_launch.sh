#!/bin/bash
set -e
# use default values
python -m paddle.distributed.launch multi_process.py

# use paddlecloud
cluster_node_ips="10.0.0.1"
node_ip="10.0.0.1"
export PADDLE_TRAINERS_NUM=2
export POD_IP=127.0.0.1
export PADDLE_TRAINERS=127.0.0.1,127.0.0.2
export PADDLE_TRAINER_ID=0

distributed_args="--use_paddlecloud True --cluster_node_ips ${cluster_node_ips} --node_ip ${node_ip} --selected_gpus=0,1 --log_dir testlog"
python -m paddle.distributed.launch ${distributed_args} multi_process.py

str1="selected_gpus:0 worker_endpoints:127.0.0.1:6170,127.0.0.1:6171,127.0.0.2:6170,127.0.0.2:6171 trainers_num:4 current_endpoint:127.0.0.1:6170 trainer_id:0"
str2="selected_gpus:1 worker_endpoints:127.0.0.1:6170,127.0.0.1:6171,127.0.0.2:6170,127.0.0.2:6171 trainers_num:4 current_endpoint:127.0.0.1:6171 trainer_id:1"
file_0="multi_process.check_0.log"
file_1="multi_process.check_1.log"

echo "paddlecloud params test"
if grep -q "$str1" "$file_0"; then
    echo "find trainer 0"
else
    echo "not find trainer 0"
    exit -1
fi

if grep -q "$str2" "$file_1"; then
    echo "find trainer 1"
else
    echo "not find trainer 1"
    exit -1
fi

# test async poll process
if [ -f $file_0 ]; then
    rm $file_0
fi
if [ -f $file_1 ]; then
    rm $file_1
fi

echo ""
echo "paddle.distributed.launch async poll process test"
if ! python -m paddle.distributed.launch ${distributed_args} multi_process.py abort; then
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
