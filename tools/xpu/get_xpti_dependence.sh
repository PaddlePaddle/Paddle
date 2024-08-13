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

set -ex

XPTI_URL=$1
XPTI_DIR_NAME=$2

if ! [ -n "$XPTI_URL" ]; then
  exit 0
fi

wget --no-check-certificate ${XPTI_URL} -c -q -O xpti.tar.gz
if [[ $? -ne 0  ]]; then
  echo "downloading failed: ${XPTI_URL}"
  exit 1
else
  echo "downloading ok: ${XPTI_URL}"
fi

tar -xvf xpti.tar.gz

# xpu/include/xpu already exists
cp -r ${XPTI_DIR_NAME}/include/* xpu/include/xpu
# xpu/lib already exists
cp -r ${XPTI_DIR_NAME}/so/* xpu/lib/
# copy libxpurt.so that support klprof
# commit fa894b83f9e2d1235564b93301265d8b55be5464 (HEAD -> trace)
rm xpu/lib/libxpurt.so*
cp -r ${XPTI_DIR_NAME}/runtime/libxpurt.so* xpu/lib/
