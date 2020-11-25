#!/bin/bash

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

CUR_DIR=$(dirname "${BASH_SOURCE[0]}")
CUR_DIR=$(realpath "$CUR_DIR")

# shellcheck disable=2034
PADDLE_DIR=$(realpath "$CUR_DIR/../../../")

BUILD_DOCKERFILE="$CUR_DIR/Dockerfile"

PYTHON_REQ="$PADDLE_DIR/python/requirements.txt"
UNITTEST_REQ="$PADDLE_DIR/python/unittest_py/requirements.txt"

PACKAGE_REQ="$CUR_DIR/package.txt"

image_tag(){
    CHKSUM=$(cat "$BUILD_DOCKERFILE" "$PACKAGE_REQ" "$PYTHON_REQ" "$UNITTEST_REQ"| md5sum - | cut -b-8)
    echo "$CHKSUM"
}

# shellcheck disable=2034
BUILD_TAG="$(image_tag)"
BUILD_NAME="paddle-musl-build"
BUILD_IMAGE="$BUILD_NAME:$BUILD_TAG"
