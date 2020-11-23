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

function test_launch_ps(){
    python -m paddle.distributed.fleet.launch --server_num=2 --worker_num=2 fleet_ps_training.py 2> ut.elog
    if grep -q "server are killed" ut.elog; then
        echo "test pserver launch succeed"
    else
        echo "test pserver launch failed"
        exit -1
    fi

    python -m paddle.distributed.fleet.launch --servers="127.0.0.1:6780,127.0.0.1:6781" --workers="127.0.0.1:6782,127.0.0.1:6783" fleet_ps_training.py 2> ut.elog
    if grep -q "server are killed" ut.elog; then
        echo "test pserver launch succeed"
    else
        echo "test pserver launch failed"
        exit -1
    fi

    python -m paddle.distributed.fleet.launch --servers="127.0.0.1:6780,127.0.0.1:6781" --workers="127.0.0.1,127.0.0.1" fleet_ps_training.py 2> ut.elog
    if grep -q "server are killed" ut.elog; then
        echo "test pserver launch succeed"
    else
        echo "test pserver launch failed"
        exit -1
    fi
}

function test_launch_ps_heter(){
    python -m paddle.distributed.fleet.launch --server_num=2 --worker_num=2 --heter_worker_num=2 fleet_ps_training.py 2> ut.elog
    if grep -q "server are killed" ut.elog; then
        echo "test heter pserver launch succeed"
    else
        echo "test pserver launch failed"
        exit -1
    fi
}

if [[ ${WITH_GPU} == "OFF" ]]; then
    echo "in cpu test mode"
    test_launch_ps
    exit 0
fi

test_launch_ps
test_launch_ps_heter
