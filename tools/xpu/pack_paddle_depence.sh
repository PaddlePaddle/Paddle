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

XDNN_URL=$3
XDNN_DIR_NAME=$4

XCCL_URL=$5
XCCL_DIR_NAME=$6

if [[ $# -eq 8 ]]; then
  echo "Compiling Paddle with XHPC"
  XHPC_URL=$7
  XHPC_DIR_NAME=$8
elif [[ $# -eq 7 ]]; then
  XHPC_DIR_NAME=$7
fi

BOS_PATTERN="https://baidu-kunlun-product.su.bcebos.com"

mkdir -p xpu/include/xhpc/xblas
mkdir -p xpu/include/xhpc/xfa
mkdir -p xpu/include/xpu
mkdir -p xpu/lib

function download_from_bos() {
  wget --no-check-certificate ${XRE_URL} -q -O xre.tar.gz
  tar xvf xre.tar.gz

  wget --no-check-certificate ${XDNN_URL} -q -O xdnn.tar.gz
  tar xvf xdnn.tar.gz

  wget --no-check-certificate ${XCCL_URL} -q -O xccl.tar.gz
  tar xvf xccl.tar.gz
}

function xhpc_prepare() {
    if ! [ -z ${XHPC_URL} ]; then
      echo "XHPC_URL: ${XHPC_URL}"
      wget --no-check-certificate ${XHPC_URL} -q -O xhpc.tar.gz
      tar xvf xhpc.tar.gz

      cp -r ${XHPC_DIR_NAME}/xblas/include/* xpu/include/xhpc/xblas
      cp -r ${XHPC_DIR_NAME}/xblas/so/* xpu/lib/

      cp -r ${XHPC_DIR_NAME}/xdnn/include/* xpu/include/
      cp -r ${XHPC_DIR_NAME}/xdnn/so/* xpu/lib

      cp -r ${XHPC_DIR_NAME}/xfa/include/* xpu/include/xhpc/xfa
      cp -r ${XHPC_DIR_NAME}/xfa/so/* xpu/lib/
    else
      cp -r ${XDNN_DIR_NAME}/include/xpu/* xpu/include/xpu/
      cp -r ${XDNN_DIR_NAME}/so/* xpu/lib/
    fi
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
    cp -r ${LOCAL_PATH}/$XRE_DIR_NAME/so/libxpurt* xpu/lib/

    # xccl assemble
    cp -r ${LOCAL_PATH}/$XCCL_DIR_NAME/include/* xpu/include/xpu/
    cp -r ${LOCAL_PATH}/$XCCL_DIR_NAME/so/* xpu/lib/

    # xhpc assemble
    cp -r ${LOCAL_PATH}/${XHPC_DIR_NAME}/xblas/include/* xpu/include/xhpc/xblas
    cp -r ${LOCAL_PATH}/${XHPC_DIR_NAME}/xblas/so/* xpu/lib/

    cp -r ${LOCAL_PATH}/${XHPC_DIR_NAME}/xdnn/include/* xpu/include/
    cp -r ${LOCAL_PATH}/${XHPC_DIR_NAME}/xdnn/so/* xpu/lib

    cp -r ${LOCAL_PATH}/${XHPC_DIR_NAME}/xfa/include/* xpu/include/xhpc/xfa
    cp -r ${LOCAL_PATH}/${XHPC_DIR_NAME}/xfa/so/* xpu/lib/
}

if [[ $XRE_URL != *"$BOS_PATTERN"* ]]; then
    # below is local way
    build_from="local"
    LOCAL_PATH=$(dirname "$XRE_URL")
    echo "LOCAL_PATH: ${LOCAL_PATH}"

    local_prepare
    local_assemble
else
    # below is default way
    build_from="bos"
    download_from_bos
    xhpc_prepare

    cp -r $XRE_DIR_NAME/include/xpu/* xpu/include/xpu/
    cp -r $XRE_DIR_NAME/so/libxpurt* xpu/lib/
    cp -r $XCCL_DIR_NAME/include/* xpu/include/xpu/
    cp -r $XCCL_DIR_NAME/so/* xpu/lib/
fi
