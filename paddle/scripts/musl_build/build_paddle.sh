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

# setup default arguments
BUILD_MAN="${BUILD_MAN-0}"
WITH_PRUNE_CONTAINER="${WITH_PRUNE_CONTAINER-1}"
WITH_TEST="${WITH_TEST-0}"

declare -a RUN_ARGS
if [ "$HTTP_PROXY" ]; then
    RUN_ARGS+=("--env" "HTTP_PROXY=$HTTP_PROXY")
    echo ">>> using http proxy: $HTTP_PROXY"
fi

if [ "$HTTPS_PROXY" ]; then
    RUN_ARGS+=("--env" "HTTPS_PROXY=$HTTPS_PROXY")
    echo ">>> using https proxy: $HTTPS_PROXY"
fi

echo ">>> compile paddle in docker"
echo ">>> docker image: $BUILD_IMAGE"

BUILD_ID=$(docker images -q "$BUILD_IMAGE")
if [ ! "$BUILD_ID" ]; then
    echo ">>> docker image is not existed, and try to build."
    WITH_REQUIREMENT=0 WITH_UT_REQUIREMENT=0 "$CUR_DIR/build_docker.sh"
fi

echo ">>> container name: $BUILD_CONTAINER"
echo ">>> mount paddle: $PADDLE_DIR => $MOUNT_DIR"

mkdir -p "$CCACHE_DIR"
echo ">>> ccache dir: $CCACHE_DIR"

mkdir -p "$CACHE_DIR"
echo ">>> local cache dir: $CACHE_DIR"

RUN_ARGS+=("--env" "WITH_REQUIREMENT=$MOUNT_DIR/$PYTHON_REQ")
echo ">>> install python requirement"


if [ "$BUILD_MAN" != "1" ]; then
    echo ">>> ========================================"
    echo ">>> automatic build mode"
    echo ">>> ========================================"

    BUILD_SCRIPT=$MOUNT_DIR/paddle/scripts/musl_build/build_inside.sh
    echo ">>> build script: $BUILD_SCRIPT"

    OUTPUT_DIR="output"
    mkdir -p $OUTPUT_DIR
    OUTPUT_DIR=$(realpath $OUTPUT_DIR)
    echo ">>> build output: $OUTPUT_DIR"

    if [ "$WITH_TEST" == "1" ]; then
        RUN_ARGS+=("--env" "WITH_TEST=1")
        echo ">>> run with unit test"

        RUN_ARGS+=("--env" "WITH_UT_REQUIREMENT=$MOUNT_DIR/$UNITTEST_REQ")
        echo ">>> install unit test requirement"
    fi

    for CTEST_FLAGS in $(env | grep ^CTEST_); do
        RUN_ARGS+=("--env" "$CTEST_FLAGS")
        echo ">>> ctest: $CTEST_FLAGS"
    done

     for CBUILD_FLAGS in $(env | grep ^FLAGS_); do
        RUN_ARGS+=("--env" "$CBUILD_FLAGS")
        echo ">>> flags: $CBUILD_FLAGS"
    done

    if [ "$WITH_PRUNE_CONTAINER" == "1" ]; then
        echo ">>> with prune container"
        RUN_ARGS+=("--rm")
    fi

    # shellcheck disable=2086,2068
    docker run \
        -v "$PADDLE_DIR":"$MOUNT_DIR" \
        -v "$OUTPUT_DIR":"/output" \
        -v "$CCACHE_DIR":"/root/.ccache" \
        -v "$CACHE_DIR":"/root/.cache" \
        --workdir /root \
        --network host \
        ${RUN_ARGS[*]} \
        --name "$BUILD_CONTAINER" \
        "$BUILD_IMAGE" \
        "$BUILD_SCRIPT" $@

    echo ">>> list output: $OUTPUT_DIR"
    find "$OUTPUT_DIR" -type f
else
    echo ">>> ========================================"
    echo ">>> manual build mode"
    echo ">>> ========================================"

    # shellcheck disable=2086
    docker run \
        -it \
        -v "$PADDLE_DIR":"$MOUNT_DIR" \
        -v "$CCACHE_DIR":"/root/.ccache" \
        -v "$CACHE_DIR":"/root/.cache" \
        --workdir /root \
        --network host \
        ${RUN_ARGS[*]} \
        --name "$BUILD_CONTAINER" \
        "$BUILD_IMAGE"
fi
