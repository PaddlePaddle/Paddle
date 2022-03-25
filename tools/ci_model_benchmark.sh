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


function check_whl {
    pip uninstall -y paddlepaddle_gpu
    pip install build/pr_whl/*.whl
    [ $? -ne 0 ] && echo "install paddle failed." && exit 1

    unzip -q build/pr_whl/*.whl -d /tmp/pr
    unzip -q build/dev_whl/*.whl -d /tmp/develop

    sed -i '/version.py/d' /tmp/pr/*/RECORD
    sed -i '/version.py/d' /tmp/develop/*/RECORD
    diff_whl=`diff /tmp/pr/*/RECORD /tmp/develop/*/RECORD|wc -l`
    [ $? -ne 0 ] && echo "diff paddle whl failed." && exit 1
    if [ ${diff_whl} -eq 0 ];then
        echo "paddle whl does not diff in PR-CI-Model-benchmark, so skip this ci"
        echo "ipipe_log_param_isSkipTest_model_benchmark: 1" 
        exit 0
    else
        echo "ipipe_log_param_isSkipTest_model_benchmark: 0"
    fi
}


function compile_install_paddle {
    export CUDA_ARCH_NAME=${CUDA_ARCH_NAME:-Auto}
    export PY_VERSION=3.7
    export WITH_DISTRIBUTE=ON
    export WITH_GPU=ON
    export WITH_TENSORRT=OFF
    export WITH_TESTING=OFF
    export WITH_UNITY_BUILD=ON
    check_whl
    cd /workspace/Paddle
    git clone --depth=1 https://github.com/paddlepaddle/benchmark.git
    cd benchmark
    set +x
    wget -q --no-proxy https://xly-devops.bj.bcebos.com/benchmark/new_clone/benchmark/benchmark_allgit.tar.gz
    tar xf benchmark_allgit.tar.gz
    set -x
}

function init_benchmark {
    cd /workspace/Paddle/benchmark
    git clone PaddleClas.bundle PaddleClas

}

function prepare_data {
    cd ${cache_dir}
    if [ -d "benchmark_data" ];then 
        echo -e "benchmark_data exist!"
    else
        mkdir benchmark_data && cd benchmark_data
        mkdir dataset && cd dataset
        wget --no-proxy -q https://paddle-qa.bj.bcebos.com/benchmark_data/Bert.zip 
        unzip Bert.zip
        wget --no-proxy -q https://paddle-qa.bj.bcebos.com/benchmark_data/imagenet100_data.zip
        unzip imagenet100_data.zip
    fi
}

function run_model_benchmark {
    cd /workspace/Paddle
    pip install build/pr_whl/*.whl
    cd ${cache_dir}/benchmark_data
    export data_path=${cfs_dir}/model_dataset/model_benchmark_data
    export prepare_path=${cfs_dir}/model_dataset/model_benchmark_prepare
    export BENCHMARK_ROOT=/workspace/Paddle/benchmark
    cd ${BENCHMARK_ROOT}/scripts/benchmark_ci
    bash model_ci.sh
}

case $1 in
  whl_check)
    compile_install_paddle
  ;;
  run_benchmark)
    init_benchmark
    prepare_data
    run_model_benchmark
  ;;
  run_all)
    compile_install_paddle
    init_benchmark
    prepare_data
    run_model_benchmark
  ;;
esac
