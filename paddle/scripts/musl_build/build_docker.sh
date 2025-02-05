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

# setup configure to default value
WITH_REQUIREMENT="${WITH_REQUIREMENT-0}"
WITH_UT_REQUIREMENT="${WITH_UT_REQUIREMENT-0}"
WITH_REBUILD="${WITH_REBUILD-0}"
PYTHON_VERSION="${PYTHON_VERSION-3.7}"

# exit when any command fails
set -e

function remove_image(){
    echo ">>> clean up docker images: $BUILD_IMAGE"
    docker rmi -f "$BUILD_IMAGE"
}

function prune_image(){
    HOURS="$(expr $1 '*' 24)"
    FILTER="until=${HOURS}h"

    echo ">>> prune old docker images: $FILTER"
    docker image prune -f -a --filter "$FILTER"
}

function build_image(){
    declare -a BUILD_ARGS
    BUILD_ARGS+=("--build-arg" "PYTHON_VERSION=$PYTHON_VERSION")
    echo ">>> python version: $PYTHON_VERSION"

    if [ "$HTTP_PROXY" ]; then
        BUILD_ARGS+=("--build-arg" "http_proxy=$HTTP_PROXY")
        echo ">>> using http proxy: $HTTP_PROXY"
    fi

    if [ "$HTTPS_PROXY" ]; then
        BUILD_ARGS+=("--build-arg" "https_proxy=$HTTPS_PROXY")
        echo ">>> using https proxy: $HTTPS_PROXY"
    fi

    echo ">>> with package requirement: $PACKAGE_REQ"
    PACKAGE_B64="$(base64 -w0 $PACKAGE_REQ)"
    BUILD_ARGS+=("--build-arg" package="$PACKAGE_B64")

    if [ "$WITH_REQUIREMENT" == "1" ]; then
        FULL_PYTHON_REQ="$PADDLE_DIR/$PYTHON_REQ"
        echo ">>> with python requirement: $FULL_PYTHON_REQ"

        PYTHON_B64="$(base64 -w0 $FULL_PYTHON_REQ)"
        BUILD_ARGS+=("--build-arg" requirement="$PYTHON_B64")
    fi

    if [ "$WITH_UT_REQUIREMENT" == "1" ]; then
        FULL_UT_REQ="$PADDLE_DIR/$UNITTEST_REQ"
        echo ">>> with unittest requirement: $FULL_UT_REQ"

        UT_B64="$(base64 -w0 $UNITTEST_REQ)"
        BUILD_ARGS+=("--build-arg" requirement_ut="$UT_B64")
    fi

    if [ "$WITH_PIP_INDEX" ]; then
        echo ">>> with pip index: $WITH_PIP_INDEX"
        BUILD_ARGS+=("--build-arg" pip_index="$WITH_PIP_INDEX")
    fi

    echo ">>> build docker image: $BUILD_IMAGE"
    # shellcheck disable=2086
    docker build \
        -t "$BUILD_IMAGE" \
        -f "$BUILD_DOCKERFILE" \
        --rm=false \
        --network host \
        ${BUILD_ARGS[*]} \
        $PWD
}

if [ "$WITH_PRUNE_DAYS" ]; then
    prune_image "$WITH_PRUNE_DAYS"
fi

if [ "$WITH_REBUILD" == "1" ]; then
    remove_image
fi

if [ "$ONLY_NAME" == "1" ]; then
    echo "$BUILD_IMAGE"
else
    build_image
fi
