#!/bin/bash

# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

cluster_node_ips="127.0.0.1"
export PADDLE_TRAINERS_NUM=4
export POD_IP=127.0.0.1
export PADDLE_TRAINERS=127.0.0.1
export PADDLE_TRAINER_ID=0

export PADDLE_PORT=35789
export TRAINER_PORTS_NUM=4

distributed_args="--ips=${cluster_node_ips} --ascend_npus=0,1,2,3 --log_dir=testlog"
python -m paddle.distributed.fleet.launch ${distributed_args} \
  ascend_group.py fleetascendgroup
