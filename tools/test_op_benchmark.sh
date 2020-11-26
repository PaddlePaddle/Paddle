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

set +ex

[ -z "$PADDLE_ROOT" ] && PADDLE_ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/.. && pwd)

# Paddle repo file name -> op name
declare -A PADDLE_FILENAME_OP_MAP
PADDLE_FILENAME_OP_MAP=(
  ["arg_min_max_op_base.h"]="arg_min arg_max"
  ["arg_min_max_op_base.cu.h"]="arg_min arg_max"
  ["activation_op.cu"]="leaky_relu elu sqrt square pow exp abs log"
  ["activation_op.h"]="relu leaky_relu elu sqrt square pow exp abs log"
  ["activation_op.cc"]="relu leaky_relu elu sqrt square pow exp abs log"
  ["interpolate_op.h"]="bilinear_interp nearest_interp trilinear_interp bicubic_interp linear_interp"
  ["interpolate_op.cc"]="bilinear_interp nearest_interp trilinear_interp bicubic_interp linear_interp"
  ["interpolate_op.cu"]="bilinear_interp nearest_interp trilinear_interp bicubic_interp linear_interp"
)

# Benchmark repo name -> op name
declare -A BENCHMARK_APINAME_OP_MAP
BENCHMARK_APINAME_OP_MAP=(
  ["argmin"]="arg_min"
  ["argmax"]="arg_max"
  ["cos_sim"]="cosine_similarity"
  ["elementwise_max"]="maximum"
  ["elementwise_min"]="minimum"
  ["bilinear_interp"]="interp_bilinear"
  ["nearest_interp"]="interp_nearest"
  ["trilinear_interp"]="interp_trilinear"
  ["bicubic_interp"]="interp_bicubic"
  ["linear_interp"]="interp_linear"
)

# ops that will run benchmark test
declare -A CHANGE_OP_MAP

# ops that benchmark repo has
declare -A BENCHMARK_OP_MAP

# ops that benchmark repo missing
declare -A BENCHMARK_MISS_OP_MAP

function LOG {
  echo "[$0:${BASH_LINENO[0]}] $*" >&2
}

# Load ops that will run benchmark test
function load_CHANGE_OP_MAP {
  local op_name change_file change_file_name
  for change_file in $(git diff --name-only origin/develop)
  do
    # match directory limit
    [[ "$change_file" =~ "paddle/fluid/operators/" ]] || continue
    # match file name limit
    [[ "$change_file" =~ "_op." ]] || continue
    LOG "[INFO] Found \"${change_file}\" changed."
    change_file_name=${change_file#*paddle/fluid/operators/}
    if [ -n "${PADDLE_FILENAME_OP_MAP[$change_file_name]}" ]
    then
      for op_name in ${PADDLE_FILENAME_OP_MAP[$change_file_name]}
      do
        LOG "[INFO] Load op: \"${op_name}\"."
        CHANGE_OP_MAP[${op_name}]="$change_file"
      done
    else
      change_file_name=${change_file_name##*/}
      LOG "[INFO] Load op: \"${change_file_name%_op*}\"."
      CHANGE_OP_MAP[${change_file_name%_op*}]="$change_file"
    fi
  done
  [ ${#CHANGE_OP_MAP[*]} -eq 0 ] && LOG "[INFO] No op to test, skip this ci." && exit 0
}

# Clone benchmark repo
function prepare_benchmark_environment {
  LOG "[INFO] Clone benchmark repo ..."
  git clone https://github.com/PaddlePaddle/benchmark.git
  [ $? -ne 0 ] && LOG "[FATAL] Clone benchmark repo fail." && exit -1
  LOG "[INFO] Collect api info ..."
  python benchmark/api/deploy/collect_api_info.py \
      --test_module_name tests_v2                 \
      --info_file api_info.txt >& 2
  [ $? -ne 0 ] && LOG "[FATAL] Collect api info fail." && exit -1
}

# Load ops that will
function load_BENCHMARK_OP_MAP {
  local line op_name api_name
  prepare_benchmark_environment
  for line in $(cat api_info.txt)
  do
    api_name=${line%%,*}
    if [ -n "${BENCHMARK_APINAME_OP_MAP[$api_name]}" ]
    then
      op_name=${BENCHMARK_APINAME_OP_MAP[$api_name]}
    else
      op_name=$api_name
    fi
    if [ -n "${CHANGE_OP_MAP[$op_name]}" ]
    then
      LOG "[INFO] Load benchmark settings with op \"${op_name}\"."
      BENCHMARK_OP_MAP[$op_name]=$line
    fi
  done
}

# compile and install paddlepaddle
function compile_install_paddlepaddle {
  LOG "[INFO] Compiling install package ..."
  export WITH_GPU=ON
  export WITH_AVX=ON
  export WITH_MKL=ON
  export RUN_TEST=OFF
  export WITH_PYTHON=ON
  export WITH_TESTING=OFF
  export BUILD_TYPE=Release
  export WITH_DISTRIBUTE=OFF
  export PYTHON_ABI=cp37-cp37m
  export CMAKE_BUILD_TYPE=Release
  [ -d build ] && rm -rf build
  bash paddle/scripts/paddle_build.sh build $(nproc)
  [ $? -ne 0 ] && LOG "[FATAL] compile fail." && exit 7
  LOG "[INFO] Uninstall Paddle ..."
  pip uninstall -y paddlepaddle paddlepaddle_gpu
  LOG "[INFO] Install Paddle ..."
  pip install build/python/dist/paddlepaddle_gpu-0.0.0-cp37-cp37m-linux_x86_64.whl
}

# run op benchmark test
function run_op_benchmark_test {
  [ ${#BENCHMARK_OP_MAP[*]} -eq 0 ] && return
  local logs_dir op_name branch_name api_info_file
  api_info_file="$(pwd)/api_info.txt"
  [ -f "$api_info_file" ] && rm -f $api_info_file
  for api_info in ${BENCHMARK_OP_MAP[*]}
  do
    echo "$api_info" >> $api_info_file
  done
  for branch_name in "develop" "test_pr"
  do
    git checkout $branch_name
    [ $? -ne 0 ] && LOG "[FATAL] Missing branch ${branch_name}." && exit 7
    LOG "[INFO] Now branch name is ${branch_name}."
    compile_install_paddlepaddle
    logs_dir="$(pwd)/logs-${branch_name}"
    [ -d $logs_dir ] && rm -rf $logs_dir/* || mkdir -p $logs_dir
    [ -z "$VISIBLE_DEVICES" ] && export VISIBLE_DEVICES=0
    pushd benchmark/api > /dev/null
    bash deploy/main_control.sh tests_v2 \
                                tests_v2/configs \
                                $logs_dir \
                                $VISIBLE_DEVICES \
                                "gpu" \
                                "speed" \
                                $api_info_file \
                                "paddle"
    popd > /dev/null
  done
}

# diff benchmakr result and miss op
function summary_problems {
  local op_name exit_code
  python ${PADDLE_ROOT}/tools/check_op_benchmark_result.py \
      --develop_logs_dir $(pwd)/logs-develop \
      --pr_logs_dir $(pwd)/logs-test_pr
  exit_code=$?
  for op_name in ${!CHANGE_OP_MAP[@]}
  do
    if [ -z "${BENCHMARK_OP_MAP[$op_name]}" ]
    then
      exit_code=8
      LOG "[WARNING] Missing test script of \"${op_name}\"(${CHANGE_OP_MAP[$op_name]}) in benchmark."
    fi
  done
  [ $exit_code -ne 0 ] && exit $exit_code
}

function main {
  LOG "[INFO] Start run op benchmark test ..."
  load_CHANGE_OP_MAP
  load_BENCHMARK_OP_MAP
  run_op_benchmark_test
  summary_problems
  LOG "[INFO] Op benchmark run success and no error!"
  exit 0
}

main
