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
CUDA_LIB=$9/lib/x64
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
    rm -rf MobileNetV2.inference.model.tar.gz
  fi
    # echo "MobileNetV2.inference.model.tar.gz has been downloaded."
  # else
    if [ $WIN_DETECT != "" ]; then
      wget -q -Y off http://paddle-inference-dist.bj.bcebos.com/MobileNetV2.inference.model.tar.gz
    else
      wget -q --no-proxy http://paddle-inference-dist.bj.bcebos.com/MobileNetV2.inference.model.tar.gz
    fi
    tar xzf *.tar.gz
  # fi
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
cd ..

#download custom_op_demo data
mkdir -p custom_op
cd custom_op
if [[ -e "custom_relu_infer_model.tgz" ]]; then
  echo "custom_relu_infer_model.tgz has been downloaded."
else
    wget -q https://paddle-inference-dist.bj.bcebos.com/inference_demo/custom_operator/custom_relu_infer_model.tgz
    tar xzf *.tgz
fi
cd ..

#download custom_pass_demo data
mkdir -p custom_pass
cd custom_pass
if [ ! -d resnet50 ]; then
    wget https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo/resnet50.tgz
    tar xzf resnet50.tgz
fi

# compile and test the demo
cd $current_dir
mkdir -p build
cd build
rm -rf *

# run all test cases before exit
EXIT_CODE=0

for WITH_STATIC_LIB in ON OFF; do
  if [ $(echo `uname` | grep "Win") != "" ]; then
    # TODO(wilber, T8T9): Do we still need to support windows gpu static library
    if [ $TEST_GPU_CPU == ON ] && [ $WITH_STATIC_LIB == ON ]; then
      continue
    fi
    # -----simple_on_word2vec on windows-----
    cmake .. -GNinja -DPADDLE_LIB=${inference_install_dir} \
      -DWITH_MKL=$TURN_ON_MKL \
      -DDEMO_NAME=simple_on_word2vec \
      -DWITH_GPU=$TEST_GPU_CPU \
      -DWITH_STATIC_LIB=$WITH_STATIC_LIB \
      -DMSVC_STATIC_CRT=$MSVC_STATIC_CRT \
      -DWITH_ONNXRUNTIME=$WITH_ONNXRUNTIME \
      -DCMAKE_BUILD_TYPE=Release \
      -DCUDA_LIB="$CUDA_LIB"
    ninja
    for use_gpu in $use_gpu_list; do
      ./simple_on_word2vec.exe \
        --dirname=$DATA_DIR/word2vec/word2vec.inference.model \
        --use_gpu=$use_gpu
      if [ $? -ne 0 ]; then
        echo "simple_on_word2vec use_gpu:${use_gpu} runs failed " > ${current_dir}/test_summary.txt
        EXIT_CODE=1
      fi
    done

    # --------tensorrt mobilenet on windows------
    if [ $USE_TENSORRT == ON -a $TEST_GPU_CPU == ON ]; then
      rm -rf *
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
          echo "simple_on_word2vec use_gpu:${use_gpu} runs failed " >> ${current_dir}/test_summary.txt
          EXIT_CODE=1
        fi
      done
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
        --modeldir=$DATA_DIR/MobileNetV2/MobileNetV2 \
        --data=$DATA_DIR/MobileNetV2/MobileNetV2/data.txt
      if [ $? -ne 0 ]; then
        echo "onnxruntime_mobilenet_demo runs failed " >> ${current_dir}/test_summary.txt
        EXIT_CODE=1
      fi
    fi

    # --------custom op demo on linux/mac------
    if [ $TEST_GPU_CPU == ON -a $WITH_STATIC_LIB == OFF ]; then
      rm -rf *
      CUSTOM_OPERATOR_FILES="custom_relu_op.cc;custom_relu_op.cu"
      cmake .. -DPADDLE_LIB=${inference_install_dir} \
        -DWITH_MKL=$TURN_ON_MKL \
        -DDEMO_NAME=custom_op_demo \
        -DWITH_GPU=$TEST_GPU_CPU \
        -DWITH_STATIC_LIB=OFF \
        -DUSE_TENSORRT=$USE_TENSORRT \
        -DTENSORRT_ROOT=$TENSORRT_ROOT_DIR \
        -DCUSTOM_OPERATOR_FILES=$CUSTOM_OPERATOR_FILES \
        -DWITH_ONNXRUNTIME=$WITH_ONNXRUNTIME
      make -j$(nproc)
      ./custom_op_demo \
        --modeldir=$DATA_DIR/custom_op/custom_relu_infer_model
      if [ $? -ne 0 ]; then
        echo "custom_op_demo runs failed " >> ${current_dir}/test_summary.txt
        EXIT_CODE=1
      fi
    fi

    # --------custom pass demo on linux/mac------
    if [ $TEST_GPU_CPU == ON -a $WITH_STATIC_LIB == OFF ]; then
      rm -rf *
      CUSTOM_OPERATOR_FILES="custom_relu_op.cc;custom_relu_op.cu"
      CUSTOM_PASS_FILES="custom_relu_pass.cc"
      cmake .. -DPADDLE_LIB=${inference_install_dir} \
        -DWITH_MKL=$TURN_ON_MKL \
        -DDEMO_NAME=custom_pass_demo \
        -DWITH_GPU=$TEST_GPU_CPU \
        -DWITH_STATIC_LIB=OFF \
        -DUSE_TENSORRT=$USE_TENSORRT \
        -DTENSORRT_ROOT=$TENSORRT_ROOT_DIR \
        -DCUSTOM_OPERATOR_FILES=$CUSTOM_OPERATOR_FILES \
        -DCUSTOM_PASS_FILES=${CUSTOM_PASS_FILES} \
        -DWITH_ONNXRUNTIME=$WITH_ONNXRUNTIME
      make -j$(nproc)
      ./custom_pass_demo \
        --modeldir=$DATA_DIR/custom_pass/resnet50
      if [ $? -ne 0 ]; then
        echo "custom_pass_demo runs failed " >> ${current_dir}/test_summary.txt
        EXIT_CODE=1
      fi
    fi
  fi
done

set +x

if [[ -f ${current_dir}/test_summary.txt ]];then
  echo " "
  echo "Summary demo_ci Failed Tests ..."
  echo "=====================test summary======================"
  echo "The following tests Failed: "
  cat ${current_dir}/test_summary.txt
  echo "========================================================"
  echo " "
fi

set -x

exit ${EXIT_CODE}
