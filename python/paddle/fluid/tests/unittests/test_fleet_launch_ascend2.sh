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

RANK_TABLE_FILE_NAME="rank_table_file.json"
cat > ${RANK_TABLE_FILE_NAME} <<EOF
{
    "status": "completed",
    "version": "1.0",
    "server_count": "2",
    "server_list": [
        {
            "server_id": "127.0.0.1",
            "device": [
                {
                    "device_id": "0",
                    "device_ip": "192.1.184.23",
                    "rank_id": "0"
                },
                {
                    "device_id": "1",
                    "device_ip": "192.2.21.93",
                    "rank_id": "1"
                }
            ]
        },
        {
            "server_id": "127.0.0.2",
            "device": [
                {
                    "device_id": "0",
                    "device_ip": "192.1.94.132",
                    "rank_id": "2"
                },
                {
                    "device_id": "1",
                    "device_ip": "192.2.94.30",
                    "rank_id": "3"
                }
            ]
        }
    ]
}
EOF

# set ascend rank table file env
export RANK_TABLE_FILE="${PWD}/${RANK_TABLE_FILE_NAME}"

# use paddlecloud
echo "begin test use paddlecloud"
cluster_node_ips="127.0.0.1,127.0.0.2"
export PADDLE_TRAINERS_NUM=2
export POD_IP=127.0.0.1
export PADDLE_TRAINERS=127.0.0.1,127.0.0.2
export PADDLE_TRAINER_ID=0

export PADDLE_PORT=35789
export TRAINER_PORTS_NUM=2

distributed_args="--run_mode=collective --log_dir=testlog"
python -m paddle.distributed.fleet.launch ${distributed_args} ascend_multi_process_collective.py fleetlaunchascend

str1="selected_accelerators:0 worker_endpoints:127.0.0.1:35789,127.0.0.1:35790,127.0.0.2:35789,127.0.0.2:35790 trainers_num:4 current_endpoint:127.0.0.1:35789 trainer_id:0 device_ids:0,1,0,1 device_id:0"
str2="selected_accelerators:1 worker_endpoints:127.0.0.1:35789,127.0.0.1:35790,127.0.0.2:35789,127.0.0.2:35790 trainers_num:4 current_endpoint:127.0.0.1:35790 trainer_id:1 device_ids:0,1,0,1 device_id:1"
file_0="multi_process_fleetlaunchascend.check_0.log"
file_1="multi_process_fleetlaunchascend.check_1.log"

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
