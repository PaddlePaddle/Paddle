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

FFT_URL=$1
FFT_DIR_NAME="xpufft"

if ! [ -n "$FFT_URL" ]; then
  exit 0
fi

mkdir -p xpu/include/fft

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

download_from_bos ${FFT_URL}
check_files ${FFT_DIR_NAME}/include/cufft.h ${FFT_DIR_NAME}/lib64/libcufft.so
cp -r ${FFT_DIR_NAME}/include/* xpu/include/xpu/
cp -r ${FFT_DIR_NAME}/lib64/* xpu/lib/
cp -r ${FFT_DIR_NAME}/include/* xpu/include/fft
patchelf --set-rpath '$ORIGIN/' xpu/lib/libcufft.so
patchelf --set-rpath '$ORIGIN/' xpu/lib/libcufft.so.10
patchelf --set-rpath '$ORIGIN/' xpu/lib/libcufft.so.10.7.2.91
