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

# check build mode auto/man
BUILD_AUTO=${BUILD_AUTO:-1}


declare -a ENV_ARGS
if [ "$HTTP_PROXY" ]; then
    ENV_ARGS+=("--env" "HTTP_PROXY=$HTTP_PROXY")
    echo "using http proxy: $HTTP_PROXY"
fi

if [ "$HTTPS_PROXY" ]; then
    ENV_ARGS+=("--env" "HTTPS_PROXY=$HTTPS_PROXY")
    echo "using https proxy: $HTTPS_PROXY"
fi

if [ "$PIP_INDEX" ]; then
    ENV_ARGS+=("--env" "PIP_INDEX=$PIP_INDEX")
fi

echo "compile paddle in docker"
echo "docker image: $BUILD_IMAGE"

BUILD_ID=$(docker images -q "$BUILD_IMAGE")
if [ ! "$BUILD_ID" ]; then
    echo "docker image is not existed, and try to build."

    "$CUR_DIR/build_docker.sh"
fi

BUILD_NAME="paddle-musl-build-$(date +%Y%m%d-%H%M%S)"
echo "container name: $BUILD_NAME"

MOUNT_DIR="/paddle"
echo "mount paddle: $PADDLE_DIR => $MOUNT_DIR"


if [ "$BUILD_AUTO" -eq "1" ]; then
    echo "enter automatic build mode"

    # no exit when fails
    set +e

    BUILD_SCRIPT=$MOUNT_DIR/paddle/scripts/musl_build/build_inside.sh
    echo "build script: $BUILD_SCRIPT"

    OUTPUT_DIR="output"
    mkdir -p $OUTPUT_DIR
    OUTPUT_DIR=$(realpath $OUTPUT_DIR)
    echo "build output: $OUTPUT_DIR"

    # shellcheck disable=2086,2068
    docker run \
        -v "$PADDLE_DIR":"$MOUNT_DIR" \
        -v "$OUTPUT_DIR":/output \
        --rm \
        --workdir /root \
        --network host \
        ${ENV_ARGS[*]} \
        --name "$BUILD_NAME" \
        "$BUILD_IMAGE" \
        "$BUILD_SCRIPT" $@

    echo "list output: $OUTPUT_DIR"
    ls "$OUTPUT_DIR"
else
    echo "enter manual build mode"

    # shellcheck disable=2086
    docker run \
        -it \
        -v "$PADDLE_DIR":"$MOUNT_DIR" \
        --workdir /root \
        --network host ${ENV_ARGS[*]}\
        --name "$BUILD_NAME" \
        "$BUILD_IMAGE"
fi
