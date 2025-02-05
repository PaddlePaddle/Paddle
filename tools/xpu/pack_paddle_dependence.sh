#!/bin/bash

# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
set -x

XRE_URL=$1
XRE_DIR_NAME=$2

XHPC_URL=$3
XHPC_DIR_NAME=$4

XCCL_URL=$5
XCCL_DIR_NAME=$6

WITH_XPU_XRE5=$7

mkdir -p xpu/include/xhpc/xblas
mkdir -p xpu/include/xhpc/xfa
mkdir -p xpu/include/xhpc/xpudnn
mkdir -p xpu/include/xpu
mkdir -p xpu/include/xre
mkdir -p xpu/lib

function download_from_bos() {
  local url=$1
  wget --no-check-certificate ${url} -q -O tmp.tar.gz
  if [[ $? -ne 0 ]]; then
    echo "downloading failed: ${url}"
    exit 1
  fi
  tar xvf tmp.tar.gz
  rm -f tmp.tar.gz
}

function check_files() {
  local files=("$@")
  for file in "${files[@]}";
  do
    echo "checking $file"
    if [[ ! -f $file ]]; then
        echo "checking failed: $file"
        exit 1
    else
        echo "checking ok: $file"
    fi
  done
}

function xre_prepare() {
  check_files ${XRE_DIR_NAME}/include/xpu/runtime.h ${XRE_DIR_NAME}/so/libxpurt.so
  if [ "$WITH_XPU_XRE5" -eq 1 ]; then
    check_files ${XRE_DIR_NAME}/so/libcudart.so
    check_files ${XRE_DIR_NAME}/so/libxpuml.so
  fi
  cp -r ${XRE_DIR_NAME}/include/xpu/* xpu/include/xpu/
  cp -r ${XRE_DIR_NAME}/so/* xpu/lib/
  cp -r ${XRE_DIR_NAME}/include/* xpu/include/xre
}

function xhpc_prepare() {
  check_files ${XHPC_DIR_NAME}/xdnn/include/xpu/xdnn.h ${XHPC_DIR_NAME}/xdnn/so/libxpuapi.so
  cp -r ${XHPC_DIR_NAME}/xdnn/include/* xpu/include/
  cp -r ${XHPC_DIR_NAME}/xdnn/so/libxpuapi.so xpu/lib

  if [ "$WITH_XPU_XRE5" -eq 1 ]; then
    check_files ${XHPC_DIR_NAME}/xblas/include/cublasLt.h ${XHPC_DIR_NAME}/xblas/so/libxpu_blas.so
    cp -r ${XHPC_DIR_NAME}/xblas/include/* xpu/include/xhpc/xblas
    cp -r ${XHPC_DIR_NAME}/xblas/so/libxpu_blas.so xpu/lib/

    check_files ${XHPC_DIR_NAME}/xfa/include/flash_api.h ${XHPC_DIR_NAME}/xfa/so/libxpu_flash_attention.so
    cp -r ${XHPC_DIR_NAME}/xfa/include/* xpu/include/xhpc/xfa
    cp -r ${XHPC_DIR_NAME}/xfa/so/libxpu_flash_attention.so xpu/lib/

    check_files ${XHPC_DIR_NAME}/xpudnn/include/xpudnn.h ${XHPC_DIR_NAME}/xpudnn/so/libxpu_dnn.so
    cp -r ${XHPC_DIR_NAME}/xpudnn/include/* xpu/include/xhpc/xpudnn
    cp -r ${XHPC_DIR_NAME}/xpudnn/so/libxpu_dnn.so xpu/lib/
  fi
}

function xccl_prepare() {
  check_files ${XCCL_DIR_NAME}/include/bkcl.h ${XCCL_DIR_NAME}/so/libbkcl.so
  cp -r ${XCCL_DIR_NAME}/include/* xpu/include/xpu/
  cp -r ${XCCL_DIR_NAME}/so/* xpu/lib/
  # FIXME(yangjianbang): 待bkcl增加RPATH后, 删除以下代码
  patchelf --set-rpath '$ORIGIN/' xpu/lib/libbkcl.so
}

function local_prepare() {
    # xre prepare
    if [[ ! -d ${LOCAL_PATH}/${XRE_DIR_NAME} ]]; then
        XRE_TAR_NAME=${XRE_DIR_NAME}.tar.gz
        tar -zxf  ${LOCAL_PATH}/${XRE_TAR_NAME} -C ${LOCAL_PATH}
    fi

    # xccl prepare
    if [[ ! -d ${LOCAL_PATH}/${XCCL_DIR_NAME} ]]; then
        XCCL_TAR_NAME=${XCCL_DIR_NAME}.tar.gz
        tar -zxf  ${LOCAL_PATH}/${XCCL_TAR_NAME} -C ${LOCAL_PATH}
    fi

    # xhpc prepare
    if [[ ! -d ${LOCAL_PATH}/${XHPC_DIR_NAME} ]]; then
        XHPC_TAR_NAME=${XHPC_DIR_NAME}.tar.gz
        tar -zxf  ${LOCAL_PATH}/${XHPC_TAR_NAME} -C ${LOCAL_PATH}
    fi
}

function local_assemble() {
    # xre assemble
    cp -r ${LOCAL_PATH}/$XRE_DIR_NAME/include/xpu/* xpu/include/xpu/
    cp -r ${LOCAL_PATH}/$XRE_DIR_NAME/so/* xpu/lib/
    cp -r ${LOCAL_PATH}/$XRE_DIR_NAME/include/* xpu/include/xre

    # xccl assemble
    cp -r ${LOCAL_PATH}/$XCCL_DIR_NAME/include/* xpu/include/xpu/
    cp -r ${LOCAL_PATH}/$XCCL_DIR_NAME/so/* xpu/lib/

    # xhpc assemble
    cp -r ${LOCAL_PATH}/${XHPC_DIR_NAME}/xdnn/include/* xpu/include/
    cp -r ${LOCAL_PATH}/${XHPC_DIR_NAME}/xdnn/so/libxpuapi.so xpu/lib

    if [ "$WITH_XPU_XRE5" -eq 1 ]; then
      cp -r ${LOCAL_PATH}/${XHPC_DIR_NAME}/xblas/include/* xpu/include/xhpc/xblas
      cp -r ${LOCAL_PATH}/${XHPC_DIR_NAME}/xblas/so/libxpu_blas.so xpu/lib/

      cp -r ${LOCAL_PATH}/${XHPC_DIR_NAME}/xfa/include/* xpu/include/xhpc/xfa
      cp -r ${LOCAL_PATH}/${XHPC_DIR_NAME}/xfa/so/libxpu_flash_attention.so xpu/lib/

      cp -r ${LOCAL_PATH}/${XHPC_DIR_NAME}/xpudnn/include/* xpu/include/xhpc/xpudnn
      cp -r ${LOCAL_PATH}/${XHPC_DIR_NAME}/xpudnn/so/libxpu_dnn.so xpu/lib/
      # FIXME(yangjianbang): 待bkcl增加RPATH后, 删除以下代码
      patchelf --set-rpath '$ORIGIN/' xpu/lib/libbkcl.so
    fi
}

if [[ $XRE_URL != "http"* ]]; then
    # below is local way
    build_from="local"
    LOCAL_PATH=$(dirname "$XRE_URL")
    echo "LOCAL_PATH: ${LOCAL_PATH}"

    local_prepare
    local_assemble
else
    # below is default way
    build_from="bos"
    download_from_bos ${XRE_URL}
    download_from_bos ${XHPC_URL}
    download_from_bos ${XCCL_URL}
    xre_prepare
    xhpc_prepare
    xccl_prepare
fi
