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

set -x
PADDLE_ROOT=$1
TURN_ON_MKL=$2 # use MKL or Openblas
TEST_GPU_CPU=$3 # test both GPU/CPU mode or only CPU mode
DATA_DIR=$4 # dataset
USE_TENSORRT=$5
TENSORRT_ROOT_DIR=$6 # TensorRT root dir, default to /usr
WITH_ONNXRUNTIME=$7
MSVC_STATIC_CRT=$8
inference_install_dir=${PADDLE_ROOT}/build/paddle_inference_install_dir
WIN_DETECT=$(echo `uname` | grep "Win") # detect current platform

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

mkdir -p $DATA_DIR
cd $DATA_DIR

if [ $7 == ON ]; then
  ONNXRUNTIME_LIB=${inference_install_dir}/third_party/install/onnxruntime/lib
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${ONNXRUNTIME_LIB}
  PADDLE2ONNX_LIB=${inference_install_dir}/third_party/install/paddle2onnx/lib
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${PADDLE2ONNX_LIB}
  #download model
  mkdir -p MobileNetV2
  cd MobileNetV2
  if [[ -e "MobileNetV2.inference.model.tar.gz" ]]; then
    echo "MobileNetV2.inference.model.tar.gz has been downloaded."
  else
    if [ $WIN_DETECT != "" ]; then
      wget -q -Y off http://paddle-inference-dist.bj.bcebos.com/MobileNetV2.inference.model.tar.gz
    else
      wget -q --no-proxy http://paddle-inference-dist.bj.bcebos.com/MobileNetV2.inference.model.tar.gz
    fi
    tar xzf *.tar.gz
  fi
  cd ..
fi

PREFIX=inference-vis-demos%2F
URL_ROOT=http://paddlemodels.bj.bcebos.com/${PREFIX}

# download vis_demo data
function download() {
  dir_name=$1
  mkdir -p $dir_name
  cd $dir_name
  if [[ -e "${PREFIX}${dir_name}.tar.gz" ]]; then
    echo "${PREFIX}${dir_name}.tar.gz has been downloaded."
  else
      if [ $WIN_DETECT != "" ]; then
        wget -q -Y off ${URL_ROOT}$dir_name.tar.gz
      else
        wget -q --no-proxy ${URL_ROOT}$dir_name.tar.gz
      fi
      tar xzf *.tar.gz
  fi
  cd ..
}

vis_demo_list='se_resnext50 ocr mobilenet'
for vis_demo_name in $vis_demo_list; do
  download $vis_demo_name
done

# download word2vec data
mkdir -p word2vec
cd word2vec
if [[ -e "word2vec.inference.model.tar.gz" ]]; then
  echo "word2vec.inference.model.tar.gz has been downloaded."
else
    wget -q http://paddle-inference-dist.bj.bcebos.com/word2vec.inference.model.tar.gz
    tar xzf *.tar.gz
fi

# compile and test the demo
cd $current_dir
mkdir -p build
cd build
rm -rf *

for WITH_STATIC_LIB in ON OFF; do
  if [ $(echo `uname` | grep "Win") != "" ]; then
    # TODO(wilber, T8T9): Do we still need to support windows gpu static library
    if [ $TEST_GPU_CPU == ON ] && [ $WITH_STATIC_LIB == ON ]; then
      continue
    fi
    # -----simple_on_word2vec on windows-----
    cmake .. -G "Visual Studio 15 2017" -A x64 -T host=x64 -DPADDLE_LIB=${inference_install_dir} \
      -DWITH_MKL=$TURN_ON_MKL \
      -DDEMO_NAME=simple_on_word2vec \
      -DWITH_GPU=$TEST_GPU_CPU \
      -DWITH_STATIC_LIB=$WITH_STATIC_LIB \
      -DMSVC_STATIC_CRT=$MSVC_STATIC_CRT \
      -DWITH_ONNXRUNTIME=$WITH_ONNXRUNTIME
    msbuild  /maxcpucount /property:Configuration=Release cpp_inference_demo.sln
    for use_gpu in $use_gpu_list; do
      Release/simple_on_word2vec.exe \
        --dirname=$DATA_DIR/word2vec/word2vec.inference.model \
        --use_gpu=$use_gpu
      if [ $? -ne 0 ]; then
        echo "simple_on_word2vec demo runs fail."
        exit 1
      fi
    done

    # -----vis_demo on windows-----
    rm -rf *
    cmake .. -G "Visual Studio 15 2017" -A x64 -T host=x64 -DPADDLE_LIB=${inference_install_dir} \
      -DWITH_MKL=$TURN_ON_MKL \
      -DDEMO_NAME=vis_demo \
      -DWITH_GPU=$TEST_GPU_CPU \
      -DWITH_STATIC_LIB=$WITH_STATIC_LIB \
      -DMSVC_STATIC_CRT=$MSVC_STATIC_CRT \
      -DWITH_ONNXRUNTIME=$WITH_ONNXRUNTIME
    msbuild  /maxcpucount /property:Configuration=Release cpp_inference_demo.sln
    for use_gpu in $use_gpu_list; do
      for vis_demo_name in $vis_demo_list; do
        Release/vis_demo.exe \
          --modeldir=$DATA_DIR/$vis_demo_name/model \
          --data=$DATA_DIR/$vis_demo_name/data.txt \
          --refer=$DATA_DIR/$vis_demo_name/result.txt \
          --use_gpu=$use_gpu
        if [ $? -ne 0 ]; then
          echo "vis demo $vis_demo_name runs fail."
          exit 1
        fi
      done
    done

    # --------tensorrt mobilenet on windows------
    if [ $USE_TENSORRT == ON -a $TEST_GPU_CPU == ON ]; then
      rm -rf *
      cmake .. -G "Visual Studio 15 2017" -A x64 -T host=x64 -DPADDLE_LIB=${inference_install_dir} \
        -DWITH_MKL=$TURN_ON_MKL \
        -DDEMO_NAME=trt_mobilenet_demo \
        -DWITH_GPU=$TEST_GPU_CPU \
        -DWITH_STATIC_LIB=$WITH_STATIC_LIB \
        -DMSVC_STATIC_CRT=$MSVC_STATIC_CRT \
        -DUSE_TENSORRT=$USE_TENSORRT \
        -DTENSORRT_ROOT=$TENSORRT_ROOT_DIR \
        -DWITH_ONNXRUNTIME=$WITH_ONNXRUNTIME
      msbuild  /maxcpucount /property:Configuration=Release cpp_inference_demo.sln
      Release/trt_mobilenet_demo.exe \
        --modeldir=$DATA_DIR/mobilenet/model \
        --data=$DATA_DIR/mobilenet/data.txt \
        --refer=$DATA_DIR/mobilenet/result.txt 
      if [ $? -ne 0 ]; then
        echo "trt demo trt_mobilenet_demo runs fail."
        exit 1
      fi
    fi
  else
    # -----simple_on_word2vec on linux/mac-----
    rm -rf *
    cmake .. -DPADDLE_LIB=${inference_install_dir} \
      -DWITH_MKL=$TURN_ON_MKL \
      -DDEMO_NAME=simple_on_word2vec \
      -DWITH_GPU=$TEST_GPU_CPU \
      -DWITH_STATIC_LIB=$WITH_STATIC_LIB \
      -DWITH_ONNXRUNTIME=$WITH_ONNXRUNTIME
    make -j$(nproc)
    word2vec_model=$DATA_DIR'/word2vec/word2vec.inference.model'
    if [ -d $word2vec_model ]; then
      for use_gpu in $use_gpu_list; do
        ./simple_on_word2vec \
          --dirname=$DATA_DIR/word2vec/word2vec.inference.model \
          --use_gpu=$use_gpu
        if [ $? -ne 0 ]; then
          echo "simple_on_word2vec demo runs fail."
          exit 1
        fi
      done
    fi
    # ---------vis_demo on linux/mac---------
    rm -rf *
    cmake .. -DPADDLE_LIB=${inference_install_dir} \
      -DWITH_MKL=$TURN_ON_MKL \
      -DDEMO_NAME=vis_demo \
      -DWITH_GPU=$TEST_GPU_CPU \
      -DWITH_STATIC_LIB=$WITH_STATIC_LIB \
      -DWITH_ONNXRUNTIME=$WITH_ONNXRUNTIME
    make -j$(nproc)
    for use_gpu in $use_gpu_list; do
      for vis_demo_name in $vis_demo_list; do
        ./vis_demo \
          --modeldir=$DATA_DIR/$vis_demo_name/model \
          --data=$DATA_DIR/$vis_demo_name/data.txt \
          --refer=$DATA_DIR/$vis_demo_name/result.txt \
          --use_gpu=$use_gpu
        if [ $? -ne 0 ]; then
          echo "vis demo $vis_demo_name runs fail."
          exit 1
        fi
      done
    done
    # --------tensorrt mobilenet on linux/mac------
    if [ $USE_TENSORRT == ON -a $TEST_GPU_CPU == ON ]; then
      rm -rf *
      cmake .. -DPADDLE_LIB=${inference_install_dir} \
        -DWITH_MKL=$TURN_ON_MKL \
        -DDEMO_NAME=trt_mobilenet_demo \
        -DWITH_GPU=$TEST_GPU_CPU \
        -DWITH_STATIC_LIB=$WITH_STATIC_LIB \
        -DUSE_TENSORRT=$USE_TENSORRT \
        -DTENSORRT_ROOT=$TENSORRT_ROOT_DIR \
        -DWITH_ONNXRUNTIME=$WITH_ONNXRUNTIME
      make -j$(nproc)
      ./trt_mobilenet_demo \
        --modeldir=$DATA_DIR/mobilenet/model \
        --data=$DATA_DIR/mobilenet/data.txt \
        --refer=$DATA_DIR/mobilenet/result.txt 
      if [ $? -ne 0 ]; then
        echo "trt demo trt_mobilenet_demo runs fail."
        exit 1
      fi
    fi

    # --------onnxruntime mobilenetv2 on linux/mac------
    if [ $WITH_ONNXRUNTIME == ON ]; then
      rm -rf *
      cmake .. -DPADDLE_LIB=${inference_install_dir} \
        -DWITH_MKL=$TURN_ON_MKL \
        -DDEMO_NAME=onnxruntime_mobilenet_demo \
        -DWITH_GPU=$TEST_GPU_CPU \
        -DWITH_STATIC_LIB=$WITH_STATIC_LIB \
        -DUSE_TENSORRT=$USE_TENSORRT \
        -DTENSORRT_ROOT=$TENSORRT_ROOT_DIR \
        -DWITH_ONNXRUNTIME=$WITH_ONNXRUNTIME
      make -j$(nproc)
      ./onnxruntime_mobilenet_demo \
        --modeldir=$DATA_DIR/MobileNetV2/MobileNetV2
      if [ $? -ne 0 ]; then
        echo "onnxruntime demo onnxruntime_mobilenet_demo runs fail."
        exit 1
      fi
    fi
  fi
done
set +x
