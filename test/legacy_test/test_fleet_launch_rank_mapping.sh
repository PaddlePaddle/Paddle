#!/bin/bash

# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

# use single node
echo "begin test"

RANK_MAPPING_FILE_NAME="rank_mapping_file.json"
cat > ${RANK_MAPPING_FILE_NAME} <<EOF
{
    "ip_ranks": [
        {
            "ip": "127.0.0.1",
            "ranks": [0,1]
        }
    ]
}
EOF

export FLAGS_START_PORT=35789
distributed_args="--rank_mapping_file ${RANK_MAPPING_FILE_NAME} --enable_auto_mapping true --log_dir=testlog"
CUDA_VISIBLE_DEVICES=0,1 python -m paddle.distributed.fleet.launch ${distributed_args} multi_process.py fleetlaunchcloud_rank_mapping

str1="selected_gpus:0 worker_endpoints:127.0.0.1:35789,127.0.0.1:35790 trainers_num:2 current_endpoint:127.0.0.1:35789 trainer_id:0"
str2="selected_gpus:1 worker_endpoints:127.0.0.1:35789,127.0.0.1:35790 trainers_num:2 current_endpoint:127.0.0.1:35790 trainer_id:1"
file_0="multi_process_fleetlaunchcloud_rank_mapping.check_0.log"
file_1="multi_process_fleetlaunchcloud_rank_mapping.check_1.log"

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
