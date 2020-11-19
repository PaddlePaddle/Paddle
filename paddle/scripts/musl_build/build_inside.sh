#!/bin/sh

# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

PADDLE_DIR=/paddle
BUILD_DIR=$PWD

echo "paddle: $PADDLE_DIR"
echo "python: $PYTHON_VERSION"

# exit when any command fails
set -e

if [ "$HTTP_PROXY" ]; then 
    echo "http_proxy: $HTTP_PROXY" 
    git config --global http.proxy "$HTTP_PROXY"
fi

if [ "$HTTP_PROXY" ]; then 
    echo "https_proxy: $HTTPS_PROXY" 
    git config --global https.proxy "$HTTPS_PROXY"
fi

echo "configure with cmake"
cmake "$PADDLE_DIR" \
    -DWITH_MUSL=ON \
    -DWITH_CRYPTO=OFF \
    -DWITH_MKL=OFF \
    -DWITH_GPU=OFF

echo "compile with make: $*"
# shellcheck disable=2068
make $@

echo "save python dist directory to /output"
cp -r python/dist /output/
