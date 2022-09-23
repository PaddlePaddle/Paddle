#!/bin/bash

# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

# test use DISTRIBUTED_TRAINER_ENDPOINTS env in paddlecloud
unset PADDLE_PORT
export DISTRIBUTED_TRAINER_ENDPOINTS=127.0.0.1:6170,127.0.0.1:6171,127.0.0.2:6170,127.0.0.2:6171
export cluster_node_ips="127.0.0.1,127.0.0.2"
export PADDLE_TRAINERS_NUM=2
export POD_IP=127.0.0.1
export PADDLE_TRAINERS=127.0.0.1,127.0.0.2
export PADDLE_TRAINER_ID=0

export TRAINER_PORTS_NUM=2

file_0="multi_process_fullpath_launch.check_0.log"
file_1="multi_process_fullpath_launch.check_1.log"

distributed_args="--ips=${cluster_node_ips} --mlus=0,1 --log_dir=testlog"

echo "paddle.distributed.fleet.launch async poll process test"
if ! MLU_VISIBLE_DEVICES=0,1 python -m paddle.distributed.fleet.launch ${distributed_args} multi_process_mlu.py fullpath_launch abort; then
    echo "train abort as planned"
fi

abort_str1="abort>>> selected_mlus:0 worker_endpoints:127.0.0.1:6170,127.0.0.1:6171,127.0.0.2:6170,127.0.0.2:6171 trainers_num:4 current_endpoint:127.0.0.1:6170 trainer_id:0"

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
    rm $file_1
    exit -1
fi

if [ -f $file_0 ]; then
    rm $file_0
fi
