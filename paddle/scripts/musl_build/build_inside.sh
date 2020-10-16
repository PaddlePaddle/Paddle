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
echo "http_proxy: $HTTP_PROXY"
echo "https_proxy: $HTTPS_PROXY"

# exit when any command fails
set -e

echo "create build dir: $BUILD_DIR"
mkdir -p "$BUILD_DIR"

if [ "$HTTP_PROXY" ]; then
    git config --global http.proxy "$HTTP_PROXY"
fi

if [ "$HTTP_PROXY" ]; then
    git config --global https.proxy "$HTTPS_PROXY"
fi

PIP_ARGS=""
if [ "$PIP_INDEX" ]; then
    PIP_DOMAIN=$(echo "$PIP_INDEX" | awk -F/ '{print $3}')
    PIP_ARGS="-i $PIP_INDEX --trusted-host $PIP_DOMAIN"
    echo "pip index: $PIP_INDEX"
fi

PYTHON_REQS=$PADDLE_DIR/python/requirements.txt
echo "install python requirements: $PYTHON_REQS"

# shellcheck disable=2086
pip install $PIP_ARGS --timeout 300 --no-cache-dir -r $PYTHON_REQS

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
