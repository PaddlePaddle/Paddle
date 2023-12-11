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

xhpc_base_url=$1
xhpc_dir_name=$2

echo "xhpc_base_url: $xhpc_base_url"
echo "xhpc_dir_name: $xhpc_dir_name"

if ! [ -n "$WITH_XPU_XHPC" ]; then
  exit 0
fi

wget --no-check-certificate ${xhpc_base_url} -c -q -O xphc.tar.gz
if [[ $? -ne 0  ]]; then
  echo "downloading failed: ${xhpc_base_url}"
  exit 1
else
  echo "downloading ok: ${xhpc_base_url}"
fi

tar -xf xphc.tar.gz

mkdir -p xpu/include/xhpc/xblas
mkdir -p xpu/include/xhpc/xfa

cp -r output/${xhpc_dir_name}/xblas/include/* xpu/include/xhpc/xblas
cp -r output/${xhpc_dir_name}/xblas/so/* xpu/lib/

cp -r output/${xhpc_dir_name}/xdnn/include/* xpu/include/
cp -r output/${xhpc_dir_name}/xdnn/so/* xpu/lib

cp -r output/${xhpc_dir_name}/xfa/include/* xpu/include/xhpc/xfa
cp -r output/${xhpc_dir_name}/xfa/so/* xpu/lib/
