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


function compile_install_paddle {
    export CUDA_ARCH_NAME=Auto
    export PY_VERSION=3.7
    export WITH_DISTRIBUTE=OFF
    export WITH_GPU=ON
    export WITH_TENSORRT=OFF
    export WITH_TESTING=OFF
    export WITH_UNITY_BUILD=ON
    bash -x paddle/scripts/paddle_build.sh build
    [ $? -ne 0 ] && echo "build paddle failed." && exit 1
    pip uninstall -y paddlepaddle_gpu
    pip install build/python/dist/paddlepaddle_gpu-0.0.0-cp37-cp37m-linux_x86_64.whl
    [ $? -ne 0 ] && echo "install paddle failed." && exit 1
}

function prepare_data {
    cd ${cache_dir}
    if [ -d "benchmark_data" ];then 
        echo -e "benchmark_data exist!"
    else
        mkdir benchmark_data
        cd benchmark_data
        mkdir dataset
        cd dataset
        wget --no-proxy -q https://paddle-qa.bj.bcebos.com/benchmark_data/Bert.zip 
        unzip Bert.zip
        wget --no-proxy -q https://paddle-qa.bj.bcebos.com/benchmark_data/imagenet100_data.zip
        unzip imagenet100_data.zip
    fi
}

function run_model_benchmark {
    cd ${cache_dir}/benchmark_data
    if [ -d "benchmark" ];then rm -rf benchmark
    fi
    git clone --recurse-submodules=PaddleClas --recurse-submodules=PaddleNLP https://github.com/paddlepaddle/benchmark.git
    export data_path=${cache_dir}/benchmark_data/dataset
    export BENCHMARK_ROOT=${cache_dir}/benchmark_data/benchmark
    cd ${BENCHMARK_ROOT}/scripts/benchmark_ci
    bash model_ci.sh
}

compile_install_paddle
prepare_data
run_model_benchmark
