#!/usr/bin/env bash

# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

function start_build_docker() {
    docker pull $IMG

    apt_mirror='s#http://archive.ubuntu.com/ubuntu#mirror://mirrors.ubuntu.com/mirrors.txt#g'
    DOCKER_ENV=$(cat <<EOL
        -e FLAGS_fraction_of_gpu_memory_to_use=0.15 \
        -e CTEST_OUTPUT_ON_FAILURE=1 \
        -e CTEST_PARALLEL_LEVEL=1 \
        -e APT_MIRROR=${apt_mirror} \
        -e WITH_GPU=ON \
        -e CUDA_ARCH_NAME=Auto \
        -e WITH_AVX=ON \
        -e WITH_GOLANG=OFF \
        -e WITH_TESTING=ON \
        -e WITH_C_API=OFF \
        -e WITH_COVERAGE=ON \
        -e COVERALLS_UPLOAD=ON \
        -e WITH_DEB=OFF \
        -e CMAKE_BUILD_TYPE=RelWithDebInfo \
        -e PADDLE_FRACTION_GPU_MEMORY_TO_USE=0.15 \
        -e CUDA_VISIBLE_DEVICES=0,1 \
        -e WITH_DISTRIBUTE=ON \
        -e WITH_FLUID_ONLY=ON \
        -e RUN_TEST=ON
EOL
    )

    DOCKER_CMD="nvidia-docker"
    if ! [ -x "$(command -v ${DOCKER_CMD})" ]; then
        DOCKER_CMD="docker"
    fi
    if [ ! -d "${HOME}/.ccache" ]; then
        mkdir ${HOME}/.ccache
    fi
    set -ex
    ${DOCKER_CMD} run -it \
        ${DOCKER_ENV} \
        -e SCRIPT_NAME=$0 \
        -v $PADDLE_ROOT:/paddle \
        -v ${HOME}/.ccache:/root/.ccache \
        -w /paddle \
        $IMG \
        paddle/scripts/paddle_build.sh $@
    set +x
}

function main() {
    DOCKER_REPO="paddlepaddle/paddle"
    VERSION="latest-dev"
    PADDLE_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}")/../../" && pwd )"
    if [ "$1" == "build_android" ]; then
        VERSION="latest-dev-android"
    fi
    IMG=${DOCKER_REPO}:${VERSION}
    start_build_docker $@
}

main $@
