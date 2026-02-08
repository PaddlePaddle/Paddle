#!/bin/bash

# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

set -ex

XPUTX_URL=$1
XPUTX_DIR_NAME=$2

if ! [ -n "$XPUTX_URL" ]; then
  exit 0
fi

wget --no-check-certificate ${XPUTX_URL} -c -q -O xprofiler.tar.gz
if [[ $? -ne 0  ]]; then
  echo "downloading failed: ${XPUTX_URL}"
  exit 1
else
  echo "downloading ok: ${XPUTX_URL}"
fi

NVTX_URL="https://klx-sdk-release-public.su.bcebos.com/nvtx3/nvtx3.tar.gz"
wget --no-check-certificate ${NVTX_URL} -c -q -O nvtx3.tar.gz
if [[ $? -ne 0  ]]; then
  echo "downloading failed: ${NVTX_URL}"
  exit 1
else
  echo "downloading ok: ${NVTX_URL}"
fi

tar -xvf xprofiler.tar.gz
tar -xvf nvtx3.tar.gz

# xpu/include/xpu already exists
cp -r nvtx3 xpu/include/
# xpu/lib already exists
cp ${XPUTX_DIR_NAME}/so/libxpuToolsExt.so xpu/lib/
