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

export RED='\033[0;31m' # red color
export NC='\033[0m' # no color
export YELLOW='\033[33m' # yellow color

cd `dirname $0`
current_dir=`pwd`
build_dir=${current_dir}/build
log_dir=${current_dir}/log
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

ocr_download_list='ocr_det_mv3_db'
for model_name in $ocr_download_list; do
    url_prefix="https://paddle-qa.bj.bcebos.com/inference_model/2.1.1/ocr"
    download $url_prefix $model_name
done

clas_download_list='LeViT'
for model_name in $clas_download_list; do
    url_prefix="https://paddle-qa.bj.bcebos.com/inference_model/2.1.1/class"
    download $url_prefix $model_name
done

nlp_download_list='ernie_text_cls'
for model_name in $nlp_download_list; do
    url_prefix="https://paddle-qa.bj.bcebos.com/inference_model/2.1.1/nlp"
    download $url_prefix $model_name
done

det_download_list='yolov3 ppyolo_mbv3 ppyolov2_r50vd'
for model_name in $det_download_list; do
    url_prefix="https://paddle-qa.bj.bcebos.com/inference_model/2.1.1/detection"
    download $url_prefix $model_name
done

unknown_download_list='resnet50_quant'
for model_name in $unknown_download_list; do
    url_prefix="https://paddle-qa.bj.bcebos.com/inference_model/unknown"
    download $url_prefix $model_name
done

function compile_test() {
    mkdir -p ${build_dir}
    cd ${build_dir}
    TEST_NAME=$1
    cmake .. -DPADDLE_LIB=${inference_install_dir} \
             -DWITH_MKL=$TURN_ON_MKL \
             -DDEMO_NAME=${TEST_NAME} \
             -DWITH_GPU=$TEST_GPU_CPU \
             -DWITH_STATIC_LIB=OFF \
             -DUSE_TENSORRT=$USE_TENSORRT \
             -DTENSORRT_ROOT=$TENSORRT_ROOT_DIR \
             -DWITH_GTEST=ON
    make -j$(nproc)
    cd -
}


# compile and run test
cd $current_dir
mkdir -p ${build_dir}
mkdir -p ${log_dir}
cd ${build_dir}
rm -rf *

# ---------tensorrt gpu tests on linux---------
if [ $USE_TENSORRT == ON -a $TEST_GPU_CPU == ON ]; then
    rm -rf *

    printf "${YELLOW} start test_resnet50 ${NC} \n";
    compile_test "test_resnet50"
    ./test_resnet50 \
        --modeldir=$DATA_DIR/resnet50/resnet50 \
        --gtest_output=xml:test_resnet50.xml
    if [ $? -ne 0 ]; then
        echo "test_resnet50 runs failed" >> ${current_dir}/build/test_summary.txt
        EXIT_CODE=1
    fi

    printf "${YELLOW} start test_det_mv3_db ${NC} \n";
    compile_test "test_det_mv3_db"
    make -j$(nproc)
    ./test_det_mv3_db \
        --modeldir=$DATA_DIR/ocr_det_mv3_db/ocr_det_mv3_db \
        --gtest_output=xml:test_det_mv3_db.xml
    if [ $? -ne 0 ]; then
        echo "test_det_mv3_db runs failed" >> ${current_dir}/build/test_summary.txt
        EXIT_CODE=1
    fi

    printf "${YELLOW} start test_LeViT ${NC} \n";
    compile_test "test_LeViT"
    ./test_LeViT \
        --modeldir=$DATA_DIR/LeViT/LeViT \
        --gtest_output=xml:test_LeViT.xml
    if [ $? -ne 0 ]; then
        echo "test_LeViT runs failed" >> ${current_dir}/build/test_summary.txt
        EXIT_CODE=1
    fi

    printf "${YELLOW} start test_ernie_text_cls ${NC} \n";
    compile_test "test_ernie_text_cls"
    ./test_ernie_text_cls \
        --modeldir=$DATA_DIR/ernie_text_cls/ernie_text_cls \
        --gtest_output=xml:test_ernie_text_cls.xml
    if [ $? -ne 0 ]; then
        echo "test_ernie_text_cls runs failed" >> ${current_dir}/build/test_summary.txt
        EXIT_CODE=1
    fi

    printf "${YELLOW} start test_yolov3 ${NC} \n";
    compile_test "test_yolov3"
    ./test_yolov3 \
        --modeldir=$DATA_DIR/yolov3/yolov3 \
        --gtest_output=xml:test_yolov3.xml
    if [ $? -ne 0 ]; then
        echo "test_yolov3 runs failed" >> ${current_dir}/build/test_summary.txt
        EXIT_CODE=1
    fi

    printf "${YELLOW} start test_ppyolo_mbv3 ${NC} \n";
    compile_test "test_ppyolo_mbv3"
    ./test_ppyolo_mbv3 \
        --modeldir=$DATA_DIR/ppyolo_mbv3/ppyolo_mbv3 \
        --gtest_output=xml:test_ppyolo_mbv3.xml
    if [ $? -ne 0 ]; then
        echo "test_ppyolo_mbv3 runs failed" >> ${current_dir}/build/test_summary.txt
        EXIT_CODE=1
    fi

    printf "${YELLOW} start test_ppyolov2_r50vd ${NC} \n";
    compile_test "test_ppyolov2_r50vd"
    ./test_ppyolov2_r50vd \
        --modeldir=$DATA_DIR/ppyolov2_r50vd/ppyolov2_r50vd \
        --gtest_output=xml:test_ppyolov2_r50vd.xml
    if [ $? -ne 0 ]; then
        echo "test_ppyolov2_r50vd runs failed" >> ${current_dir}/build/test_summary.txt
        EXIT_CODE=1
    fi

    printf "${YELLOW} start test_resnet50_quant ${NC} \n";
    compile_test "test_resnet50_quant"
    ./test_resnet50_quant \
        --int8dir=$DATA_DIR/resnet50_quant/resnet50_quant/resnet50_quant \
        --modeldir=$DATA_DIR/resnet50/resnet50 \
        --datadir=$DATA_DIR/resnet50_quant/resnet50_quant/imagenet-eval-binary/9.data \
        --gtest_output=xml:test_resnet50_quant.xml
    if [ $? -ne 0 ]; then
        echo "test_resnet50_quant runs failed" >> ${current_dir}/build/test_summary.txt
        EXIT_CODE=1
    fi

    cp ./*.xml ${log_dir};
fi


if [[ -f ${current_dir}/build/test_summary.txt ]];then
  echo "=====================test summary======================"
  cat ${current_dir}/build/test_summary.txt
  echo "========================================================"
fi
echo "infer_ut script finished"
exit ${EXIT_CODE}
