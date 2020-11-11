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

# shellcheck disable=1090
source "$CUR_DIR/config.sh"

# exit when any command fails
set -e

declare -a ENV_ARGS
if [ "$HTTP_PROXY" ]; then
    ENV_ARGS+=("--build-arg" "http_proxy=$HTTP_PROXY")
    echo "using http proxy: $HTTP_PROXY"
fi

if [ "$HTTPS_PROXY" ]; then
    ENV_ARGS+=("--build-arg" "https_proxy=$HTTPS_PROXY")
    echo "using https proxy: $HTTPS_PROXY"
fi

echo "clean up docker images: $BUILD_IMAGE"
docker rmi -f "$BUILD_IMAGE"

echo "build docker image: $BUILD_IMAGE"

# shellcheck disable=2086
docker build \
    -t "$BUILD_IMAGE" \
    -f "$CUR_DIR/Dockerfile" \
    --rm=false \
    --network host \
    ${ENV_ARGS[*]} \
    --output type=tar,dest=build.tar \
    .
