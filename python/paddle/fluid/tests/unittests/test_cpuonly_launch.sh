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

function test_launch_cpuonly(){
    python -m paddle.distributed.launch --nproc_per_node=4  \
        parallel_dygraph_gradient_check.py 2>ut.elog
    if grep -q "ABORT" ut.elog; then
        echo "test cpu only failed"
        exit -1
    else
        if grep -q "CPUONLY" ut.elog; then
            echo "test cpu only successfully"
        else 
            echo "test cpu only failed"
            exit -1
        fi
    fi
}

echo "in out loop test mode"
test_launch_cpuonly
