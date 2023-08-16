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

wget --no-check-certificate ${XRE_URL} -c -q -O xre.tar.gz
tar xvf xre.tar.gz

wget --no-check-certificate ${XDNN_URL} -c -q -O xdnn.tar.gz
tar xvf xdnn.tar.gz

wget --no-check-certificate ${XCCL_URL} -c -q -O xccl.tar.gz
tar xvf xccl.tar.gz

mkdir -p xpu/include/xpu
mkdir -p xpu/lib

cp -r $XRE_DIR_NAME/include/xpu/* xpu/include/xpu/
cp -r $XRE_DIR_NAME/so/libxpurt* xpu/lib/
cp -r $XDNN_DIR_NAME/include/xpu/* xpu/include/xpu/
cp -r $XDNN_DIR_NAME/so/libxpuapi.so xpu/lib/
cp -r $XCCL_DIR_NAME/include/* xpu/include/xpu/
cp -r $XCCL_DIR_NAME/so/* xpu/lib/
