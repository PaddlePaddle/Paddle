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

partition_name=pod64
vipu_server=10.137.96.62
allclose_script="
import sys
import numpy as np
data1 = np.loadtxt(\"ipu_res.txt\")
data2 = np.loadtxt(\"cpu_res.txt\")
if np.allclose(data1[::16], data2, atol=1e-6):
    sys.exit(0)
else:
    sys.exit(1)
"

for opt in lamb sgd adam ;
do
    for onchip in False True ;
    do
        for rts in False True ;
        do
            echo "Testcase: opt: ${opt}, onchip: ${onchip}, rts: ${rts}"
            echo "paddle.distributed.fleet.launch test with IPUs..."
            python3.8 -m paddle.distributed.launch \
            --devices=8 \
            ipu \
            --hosts=localhost \
            --nproc_per_host=2 \
            --ipus_per_replica=2 \
            --ipu_partition=${partition_name} \
            --vipu_server=${vipu_server} \
            test_dist_data_parallel_ipu.py ${opt} ipu_res.txt ${onchip} ${rts} > ipu.log
            echo "paddle.distributed.fleet.launch test with IPUs...Done"

            echo "paddle normal test with CPU..."
            export POPLAR_IPUMODEL=1
            python3.8 test_dist_data_parallel_ipu.py ${opt} cpu_res.txt > cpu.log
            unset POPLAR_IPUMODEL
            echo "paddle normal test with CPU...Done"

            echo "Compare results..."
            python3.8 -c """${allclose_script}"""
            if [ $? -eq 0 ];then
            echo "Compare results...Done"
            else
            echo "Error occurs. Please check ipu.log, cpu.log, ipu_res.txt and cpu_res.txt"
            exit 0
            fi
        done
    done
done

if [ -f "ipu.log" ]; then
    rm "ipu.log"
fi
if [ -f "cpu.log" ]; then
    rm "cpu.log"
fi
if [ -f "ipu_res.txt" ]; then
    rm "ipu_res.txt"
fi
if [ -f "cpu_res.txt" ]; then
    rm "cpu_res.txt"
fi
