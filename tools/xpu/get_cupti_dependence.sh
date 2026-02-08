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

CUPTI_URL=$1
CUPTI_DIR_NAME=$2

if ! [ -n "$CUPTI_URL" ]; then
  exit 0
fi

wget --no-check-certificate ${CUPTI_URL} -c -q -O cupti.tar.gz
if [[ $? -ne 0  ]]; then
  echo "downloading failed: ${CUPTI_URL}"
  exit 1
else
  echo "downloading ok: ${CUPTI_URL}"
fi

tar -xvf cupti.tar.gz

# xpu/include/xpu already exists
cp -r ${CUPTI_DIR_NAME}/include/cupti xpu/include/
# xpu/lib already exists
cp ${CUPTI_DIR_NAME}/lib/* xpu/lib/
