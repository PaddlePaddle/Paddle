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

export PYTHONPATH=/paddle/working/github/windstamp/Paddle/build/python:$PYTHONPATH

RANK_TABLE_FILE_NAME="rank_table_file.json"
cat > ${RANK_TABLE_FILE_NAME} <<EOF
{
    "status": "completed",
    "version": "1.0",
    "server_count": "1",
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
        }
    ]
}
EOF

# set ascend rank table file env
export RANK_TABLE_FILE="${PWD}/${RANK_TABLE_FILE_NAME}"

# use ascend
echo "begin test use ascend npu"

distributed_args="--run_mode=collective --log_dir=testlog"
python -m paddle.distributed.fleet.launch ${distributed_args} npu/test_sync_batch_norm_op_npu_3.py
