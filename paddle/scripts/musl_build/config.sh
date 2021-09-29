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
MOUNT_DIR="/paddle"

BUILD_DOCKERFILE="$CUR_DIR/Dockerfile"
PACKAGE_REQ="$CUR_DIR/package.txt"

PYTHON_REQ="python/requirements.txt"
UNITTEST_REQ="python/unittest_py/requirements.txt"

function chksum(){
    cat $* | md5sum - | cut -b-8
}

# shellcheck disable=2034
BUILD_NAME="paddle-musl-build"
BUILD_TAG=$(chksum "$BUILD_DOCKERFILE" "$PACKAGE_REQ")
BUILD_IMAGE="$BUILD_NAME:$BUILD_TAG"
BUILD_CONTAINER="$BUILD_NAME-$(date +%Y%m%d-%H%M%S)"

CCACHE_DIR="${CCACHE_DIR-${HOME}/.paddle-musl/ccache}"
CACHE_DIR="${CACHE_DIR-${HOME}/.paddle-musl/cache}"
