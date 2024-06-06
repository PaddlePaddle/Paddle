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

#test for random ports
file_0_0="test_launch_filelock_0_0.log"
file_1_0="test_launch_filelock_1_0.log"
rm -rf $file_0_0 $file_0_1

distributed_args="--gpus=0,1 --log_dir=testlog"
export PADDLE_LAUNCH_LOG="test_launch_filelock_0"
CUDA_VISIBLE_DEVICES=0,1 python -m paddle.distributed.fleet.launch ${distributed_args} find_ports.py
str_0="worker_endpoints:127.0.0.1:6070,127.0.0.1:6071"
