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

server_port_0=${PADDLE_DIST_UT_PORT}
server_port_1=$(( PADDLE_DIST_UT_PORT + 1 ))
worker_port_0=$(( PADDLE_DIST_UT_PORT + 2 ))
worker_port_1=$(( PADDLE_DIST_UT_PORT + 3 ))
heter_worker_port_0=$(( PADDLE_DIST_UT_PORT + 4 ))
heter_worker_port_1=$(( PADDLE_DIST_UT_PORT + 5 ))

function test_launch_ps(){

    python -m paddle.distributed.fleet.launch \
        --servers="127.0.0.1:${server_port_0},127.0.0.1:${server_port_1}" \
        --workers="127.0.0.1:${worker_port_0},127.0.0.1:${worker_port_1}" \
        fleet_ps_training.py 2> ut.elog
    if grep -q "server are killed" ut.elog; then
        echo "test pserver launch succeed"
    else
        echo "test pserver launch failed"
        exit -1
    fi
}

function test_launch_ps_heter(){
    #python -m paddle.distributed.fleet.launch --server_num=2 --worker_num=2 --heter_worker_num=2 fleet_ps_training.py 2> ut.elog
    python -m paddle.distributed.fleet.launch \
        --servers="127.0.0.1:${server_port_0},127.0.0.1:${server_port_1}" \
        --workers="127.0.0.1:${worker_port_0},127.0.0.1:${worker_port_1}" \
        --heter_workers="127.0.0.1:${heter_worker_port_0},127.0.0.1:${heter_worker_port_1}" \
        fleet_ps_training.py 2> ut.elog
    if grep -q "server are killed" ut.elog; then
        echo "test heter pserver launch succeed"
    else
        echo "test pserver launch failed"
        exit -1
    fi
}

if [[ ${WITH_GPU} == "OFF" && ("${WITH_ROCM}x" == "x" || ${WITH_ROCM} == "OFF") ]]; then
    echo "in cpu test mode"
    test_launch_ps
    exit 0
fi

test_launch_ps
test_launch_ps_heter
