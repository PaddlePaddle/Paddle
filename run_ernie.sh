#!/bin/bash -ex

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

export CUDA_VISIBLE_DEVICES=6

export LD_LIBRARY_PATH=/work/Develop/Paddle/build/third_party/install/mklml/lib/:/work/cudnn_v8.2.1/cuda/lib64/:/work/TensorRT-8.0.3.4/lib/:${LD_LIBRARY_PATH}

model_path=${1:-'./linjianhe_ernie'}
input_file=${2:-'./paddle_debug.txt'}
ernie_bin=./build/paddle/fluid/inference/tests/api/ernie_debug_tool

# export GLOG_vmodule=build_cinn_pass=4

${ernie_bin}                         \
    --model_dir=${model_path}        \
    --input_file=${input_file}       \
    --req_with_batch=true            \
    --run_batch=10                   \
    --mode=no_opt                    \
    --clear_passes=false             \
    --delete_pass=false              \
    --paddle_fp16=false              \
    --enable_cinn=true
