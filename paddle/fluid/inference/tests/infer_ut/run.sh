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

set -x
PADDLE_ROOT=$1
TURN_ON_MKL=$2 # use MKL or Openblas
TEST_GPU_CPU=$3 # test both GPU/CPU mode or only CPU mode
DATA_DIR=$4 # dataset
TENSORRT_ROOT_DIR=$5 # TensorRT ROOT dir, default to /usr/local/TensorRT
MSVC_STATIC_CRT=$6
inference_install_dir=${PADDLE_ROOT}/build/paddle_inference_install_dir
EXIT_CODE=0 # init default exit code

cd `dirname $0`
current_dir=`pwd`
if [ $2 == ON ]; then
  # You can export yourself if move the install path
  MKL_LIB=${inference_install_dir}/third_party/install/mklml/lib
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${MKL_LIB}
fi
if [ $3 == ON ]; then
  use_gpu_list='true false'
else
  use_gpu_list='false'
fi

USE_TENSORRT=OFF
if [ -d "$TENSORRT_ROOT_DIR" ]; then
  USE_TENSORRT=ON
fi

# download vis_demo data
function download() {
  url_prefix=$1
  model_name=$2
  mkdir -p $model_name
  cd $model_name
  if [[ -e "${model_name}.tgz" ]]; then
    echo "${model_name}.tgz has been downloaded."
  else
      wget -q --no-proxy ${url_prefix}/${model_name}.tgz
      tar xzf *.tgz
  fi
  cd ..
}

mkdir -p $DATA_DIR
cd $DATA_DIR
download_list='resnet50'
for model_name in $download_list; do
    url_prefix="https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo"
    download $url_prefix $model_name
done

# compile and run test
cd $current_dir
mkdir -p build
cd build
rm -rf *

# ---------tensorrt resnet50 on linux---------
if [ $USE_TENSORRT == ON -a $TEST_GPU_CPU == ON ]; then
    rm -rf *
    cmake .. -DPADDLE_LIB=${inference_install_dir} \
        -DWITH_MKL=$TURN_ON_MKL \
        -DDEMO_NAME=test_resnet50 \
        -DWITH_GPU=$TEST_GPU_CPU \
        -DWITH_STATIC_LIB=OFF \
        -DUSE_TENSORRT=$USE_TENSORRT \
        -DTENSORRT_ROOT=$TENSORRT_ROOT_DIR \
        -DWITH_GTEST=ON
    make -j$(nproc)
    ./test_resnet50 \
        --modeldir=$DATA_DIR/resnet50/resnet50 \
        --gtest_output=xml:test_resnet50.xml
    if [ $? -ne 0 ]; then
        echo "test_resnet50 runs failed" >> ${current_dir}/build/test_summary.txt
        EXIT_CODE=1
    fi
fi

if [[ -f ${current_dir}/build/test_summary.txt ]];then
  echo "=====================test summary======================"
  cat ${current_dir}/build/test_summary.txt
  echo "========================================================"
fi
echo "infer_ut script finished"
exit ${EXIT_CODE}
