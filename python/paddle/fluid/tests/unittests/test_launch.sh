#!/bin/bash
set -e

node_ips="127.0.0.1"
node_id="0"
current_node_ip="127.0.0.1"

distributed_args="--node_ips ${node_ips} --node_id ${node_id} --current_node_ip ${current_node_ip}"
python -m paddle.distributed.launch ${distributed_args} multi_process.py
