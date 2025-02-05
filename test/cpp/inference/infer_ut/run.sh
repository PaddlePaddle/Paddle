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
export PADDLE_SOURCE_DIR=$1
TURN_ON_MKL=$2 # use MKL or Openblas
TEST_GPU_CPU=$3 # test both GPU/CPU mode or only CPU mode
DATA_DIR=$4 # dataset
TENSORRT_ROOT_DIR=$5 # TensorRT ROOT dir, default to /usr/local/TensorRT
WITH_ONNXRUNTIME=$6
MSVC_STATIC_CRT=$7
CUDA_LIB=$8/lib/x64
inference_install_dir=${PADDLE_ROOT}/build/paddle_inference_install_dir
EXIT_CODE=0 # init default exit code
WIN_DETECT=$(echo `uname` | grep "Win") # detect current platform
test_suite_list="cpu_tester*" # init test suite list, pass to --gtest_filter

export RED='\033[0;31m' # red color
export NC='\033[0m' # no color
export YELLOW='\033[33m' # yellow color

cd `dirname $0`
current_dir=`pwd`
build_dir=${current_dir}/build
log_dir=${current_dir}/log

# check onednn installation
if [ $2 == ON ]; then
  # You can export yourself if move the install path
  MKL_LIB=${inference_install_dir}/third_party/install/mklml/lib
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${MKL_LIB}
  test_suite_list="${test_suite_list}:mkldnn_tester*"
fi

if [ $3 == ON ]; then
  use_gpu_list='true false'
  test_suite_list="${test_suite_list}:gpu_tester*"
else
  use_gpu_list='false'
fi

# check tensorrt installation
TENSORRT_COMPILED=$(cat "${inference_install_dir}/version.txt" | grep "WITH_TENSORRT")
USE_TENSORRT=OFF
if [ -d "$TENSORRT_ROOT_DIR" ] && [ ! -z "$TENSORRT_COMPILED" ]  ; then
  USE_TENSORRT=ON
  test_suite_list="${test_suite_list}:tensorrt_tester*"
fi

function download() {
  url_prefix=$1
  model_name=$2
  mkdir -p $model_name
  cd $model_name
  if [[ -e "${model_name}.tgz" ]]; then
    echo "${model_name}.tgz has been downloaded."
  else
      if [ "$WIN_DETECT" != "" ]; then
        wget -q -Y off ${url_prefix}/${model_name}.tgz
        tar xzf *.tgz
      else
        wget -q --no-proxy ${url_prefix}/${model_name}.tgz
        tar xzf *.tgz
      fi
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

# ernie int8 quant with matmul
unknown_nlp_download_list='quant_post_model_xnli_predict_matmul'
for model_name in $unknown_nlp_download_list; do
    url_prefix="https://paddle-qa.bj.bcebos.com/inference_model/unknown/nlp"
    download $url_prefix $model_name
done

# mobilnetv1 with prune op attribute
dev_class_download_list='MobileNetV1'
for model_name in $dev_class_download_list; do
    url_prefix="https://paddle-qa.bj.bcebos.com/inference_model/2021-09-16/class"
    download $url_prefix $model_name
done

function compile_test() {
    mkdir -p ${build_dir}
    cd ${build_dir}
    TEST_NAME=$1
    if [ "$WIN_DETECT" != "" ]; then
        cmake .. -GNinja -DPADDLE_LIB=${inference_install_dir} \
             -DWITH_MKL=$TURN_ON_MKL \
             -DDEMO_NAME=${TEST_NAME} \
             -DWITH_GPU=$TEST_GPU_CPU \
             -DWITH_STATIC_LIB=OFF \
             -DUSE_TENSORRT=$USE_TENSORRT \
             -DTENSORRT_ROOT=$TENSORRT_ROOT_DIR \
             -DMSVC_STATIC_CRT=$MSVC_STATIC_CRT \
             -DWITH_GTEST=ON \
             -DCMAKE_CXX_FLAGS='/std:c++17' \
             -DCMAKE_BUILD_TYPE=Release \
             -DWITH_ONNXRUNTIME=$WITH_ONNXRUNTIME \
             -DCUDA_LIB="$CUDA_LIB"
        ninja
    else
        cmake .. -DPADDLE_LIB=${inference_install_dir} \
                 -DWITH_MKL=$TURN_ON_MKL \
                 -DDEMO_NAME=${TEST_NAME} \
                 -DWITH_GPU=$TEST_GPU_CPU \
                 -DWITH_STATIC_LIB=OFF \
                 -DUSE_TENSORRT=$USE_TENSORRT \
                 -DTENSORRT_ROOT=$TENSORRT_ROOT_DIR \
                 -DWITH_GTEST=ON \
                 -DWITH_ONNXRUNTIME=$WITH_ONNXRUNTIME
        make -j$(nproc)
    fi
    cd -
}


# compile and run test
cd $current_dir
mkdir -p ${build_dir}
mkdir -p ${log_dir}
cd ${build_dir}
rm -rf *

exe_dir=${build_dir}

# printf "${YELLOW} start test_resnet50 ${NC} \n";
# compile_test "test_resnet50"
# ${exe_dir}/test_resnet50 \
#     --modeldir=$DATA_DIR/resnet50/resnet50 \
#     --gtest_filter=${test_suite_list} \
#     --gtest_output=xml:${log_dir}/test_resnet50.xml
# if [ $? -ne 0 ]; then
#     echo "${RED} test_resnet50 runs failed ${NC}" >> ${exe_dir}/test_summary.txt
#     EXIT_CODE=8
# fi

# printf "${YELLOW} start test_det_mv3_db ${NC} \n";
# compile_test "test_det_mv3_db"
# ${exe_dir}/test_det_mv3_db \
#     --modeldir=$DATA_DIR/ocr_det_mv3_db/ocr_det_mv3_db \
#     --gtest_filter=${test_suite_list} \
#     --gtest_output=xml:${log_dir}/test_det_mv3_db.xml
# if [ $? -ne 0 ]; then
#     echo "${RED} test_det_mv3_db runs failed ${NC}" >> ${exe_dir}/test_summary.txt
#     EXIT_CODE=8
# fi

# printf "${YELLOW} start test_LeViT ${NC} \n";
# compile_test "test_LeViT"
# ${exe_dir}/test_LeViT \
#     --modeldir=$DATA_DIR/LeViT/LeViT \
#     --gtest_filter=${test_suite_list} \
#     --gtest_output=xml:${log_dir}/test_LeViT.xml
# if [ $? -ne 0 ]; then
#     echo "${RED} test_LeViT runs failed ${NC}" >> ${exe_dir}/test_summary.txt
#     EXIT_CODE=8
# fi

if [ "$WIN_DETECT" != "" ]; then
    #TODO(OliverLPH): enable test_ernie_text_cls on windows after fix compile issue
    echo "  skip test_ernie_text_cls  "
else
    printf "${YELLOW} start test_ernie_text_cls ${NC} \n";
    compile_test "test_ernie_text_cls"
    ${exe_dir}/test_ernie_text_cls \
        --modeldir=$DATA_DIR/ernie_text_cls/ernie_text_cls \
        --gtest_filter=${test_suite_list} \
        --gtest_output=xml:${log_dir}/test_ernie_text_cls.xml
    if [ $? -ne 0 ]; then
        echo "${RED} test_ernie_text_cls runs failed ${NC}" >> ${exe_dir}/test_summary.txt
        EXIT_CODE=8
    fi
fi

printf "${YELLOW} start test_yolov3 ${NC} \n";
compile_test "test_yolov3"
${exe_dir}/test_yolov3 \
    --modeldir=$DATA_DIR/yolov3/yolov3 \
    --gtest_filter=${test_suite_list} \
    --gtest_output=xml:${log_dir}/test_yolov3.xml
if [ $? -ne 0 ]; then
    echo "${RED} test_yolov3 runs failed ${NC}" >> ${exe_dir}/test_summary.txt
    EXIT_CODE=8
fi

# printf "${YELLOW} start test_ppyolo_mbv3 ${NC} \n";
# compile_test "test_ppyolo_mbv3"
# ${exe_dir}/test_ppyolo_mbv3 \
#     --modeldir=$DATA_DIR/ppyolo_mbv3/ppyolo_mbv3 \
#     --gtest_filter=${test_suite_list} \
#     --gtest_output=xml:${log_dir}/test_ppyolo_mbv3.xml
# if [ $? -ne 0 ]; then
#     echo "${RED} test_ppyolo_mbv3 runs failed ${NC}" >> ${exe_dir}/test_summary.txt
#     EXIT_CODE=8
# fi

# printf "${YELLOW} start test_ppyolov2_r50vd ${NC} \n";
# compile_test "test_ppyolov2_r50vd"
# ${exe_dir}/test_ppyolov2_r50vd \
#     --modeldir=$DATA_DIR/ppyolov2_r50vd/ppyolov2_r50vd \
#     --gtest_filter=${test_suite_list} \
#     --gtest_output=xml:${log_dir}/test_ppyolov2_r50vd.xml
# if [ $? -ne 0 ]; then
#     echo "${RED} test_ppyolov2_r50vd runs failed ${NC}" >> ${exe_dir}/test_summary.txt
#     EXIT_CODE=8
# fi

printf "${YELLOW} start test_resnet50_quant ${NC} \n";
compile_test "test_resnet50_quant"
${exe_dir}/test_resnet50_quant \
    --int8dir=$DATA_DIR/resnet50_quant/resnet50_quant/resnet50_quant \
    --modeldir=$DATA_DIR/resnet50/resnet50 \
    --datadir=$DATA_DIR/resnet50_quant/resnet50_quant/imagenet-eval-binary/9.data \
    --gtest_filter=${test_suite_list} \
    --gtest_output=xml:${log_dir}/test_resnet50_quant.xml
if [ $? -ne 0 ]; then
    echo "${RED} test_resnet50_quant runs failed ${NC}" >> ${exe_dir}/test_summary.txt
    EXIT_CODE=8
fi

# printf "${YELLOW} start test_ernie_xnli_int8 ${NC} \n";
# compile_test "test_ernie_xnli_int8"
# ernie_qat_model="quant_post_model_xnli_predict_matmul"
# ${exe_dir}/test_ernie_xnli_int8 \
#     --modeldir=$DATA_DIR/$ernie_qat_model/$ernie_qat_model \
#     --datadir=$DATA_DIR/$ernie_qat_model/$ernie_qat_model/xnli_var_len \
#     --truth_data=$DATA_DIR/$ernie_qat_model/$ernie_qat_model/truth_data \
#     --gtest_filter=${test_suite_list} \
#     --gtest_output=xml:${log_dir}/test_ernie_xnli_int8.xml
# if [ $? -ne 0 ]; then
#     echo "${RED} test_ernie_xnli_int8 runs failed ${NC}" >> ${exe_dir}/test_summary.txt
#     EXIT_CODE=8
# fi

printf "${YELLOW} start test_mobilnetv1 ${NC} \n";
compile_test "test_mobilnetv1"
${exe_dir}/test_mobilnetv1 \
    --modeldir=$DATA_DIR/MobileNetV1/MobileNetV1 \
    --gtest_filter=${test_suite_list} \
    --gtest_output=xml:${log_dir}/test_mobilnetv1.xml
if [ $? -ne 0 ]; then
    echo "${RED} test_mobilnetv1 runs failed ${NC}" >> ${exe_dir}/test_summary.txt
    EXIT_CODE=8
fi

set +x

test_suites=$(echo ${test_suite_list} | sed 's/:/ /g')
echo " "
echo "CI Tested Following Patterns: "
echo "=====================test patterns======================"
for test_suite in ${test_suites}; do
  echo "  ${test_suite}"
done
echo "========================================================"
echo " "

if [[ -f ${exe_dir}/test_summary.txt ]];then
  echo " "
  echo "Summary infer_ut Failed Tests ..."
  echo "=====================test summary======================"
  echo "The following tests Failed: "
  cat ${exe_dir}/test_summary.txt
  echo "========================================================"
  echo " "
fi
set -x

# tar Gtest output report
tar -zcvf infer_ut_log.tgz ${log_dir}

echo "infer_ut script finished"
exit ${EXIT_CODE}
