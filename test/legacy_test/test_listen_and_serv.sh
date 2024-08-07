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

unset https_proxy http_proxy

nohup python -u test_listen_and_serv_op.py > test_listen_and_serv_op.log 2>&1 &
pid=$!

flag1=test_handle_signal_in_serv_op.flag
flag2=test_list_and_serv_run_empty_optimize_block.flag

for i in {1..10}; do
    sleep 6s
    if [[ -f "${flag1}" && -f "${flag2}" ]];  then
        echo "test_listen_and_serv_op exit"
        exit 0
    fi
done

echo "test_listen_and_serv_op.log context"
cat test_listen_and_serv_op.log

#display system context
for i in {1..4}; do
    sleep 2
    top -b -n1  | head -n 50
    echo "${i}"
    top -b -n1 -i  | head -n 50
    nvidia-smi
done

#display /tmp/files
ls -l /tmp/paddle.*

if ! pgrep -x test_listen_and_serv_op; then
    exit 1
fi

kill -9 $pid

echo "after kill ${pid}"

#display system context
for i in {1..4}; do
    sleep 2
    top -b -n1  | head -n 50
    top -b -n1 -i  | head -n 50
    nvidia-smi
done

exit 1
