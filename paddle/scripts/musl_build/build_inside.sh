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
BUILD_DIR=$PWD/build

echo "paddle: $PADDLE_DIR"
echo "python: $PYTHON_VERSION"

# exit when any command fails
set -e

# setup build dir
echo "setup build dir: $BUILD_DIR"
mkdir -p $BUILD_DIR

if [ "$HTTP_PROXY" ]; then 
    echo "http_proxy: $HTTP_PROXY" 
    git config --global http.proxy "$HTTP_PROXY"
fi

if [ "$HTTP_PROXY" ]; then 
    echo "https_proxy: $HTTPS_PROXY" 
    git config --global https.proxy "$HTTPS_PROXY"
fi

BUILD_ARG=""
if [ "$WITH_TEST" == "1" ]; then
    echo "build paddle with testing"
    BUILD_ARG="-DWITH_TESTING=ON"
else
    BUILD_ARG="-DWITH_TESTING=OFF"
fi

echo "configure with cmake"
cmake "$PADDLE_DIR" \
    -DWITH_MUSL=ON \
    -DWITH_CRYPTO=OFF \
    -DWITH_MKL=OFF \
    -DWITH_GPU=OFF \
    "$BUILD_ARG"

echo "compile with make: $*"
# shellcheck disable=2068
make $@

OUTPUT_WHL="$(find python/dist/ -type f -name '*.whl'| head -n1)"
echo "paddle wheel: $OUTPUT_WHL"

echo "save paddle wheel package to /output"
cp  "$OUTPUT_WHL" /output/

if [ "$WITH_TEST" == "1" ]; then
    echo "install paddle wheel package"
    pip3 install --no-cache --force-overwrite "$OUTPUT_WHL"

    echo "run ctest"
    ctest --output-on-failure
fi
