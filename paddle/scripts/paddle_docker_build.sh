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

function container_running() {
    name=$1
    docker ps -a --format "{{.Names}}" | grep "${name}" > /dev/null
    return $?
}

function start_build_docker() {
    docker pull $IMG

    if container_running "${CONTAINER_ID}"; then
        docker stop "${CONTAINER_ID}" 1>/dev/null
        docker rm -f "${CONTAINER_ID}" 1>/dev/null
    fi

    DOCKER_ENV=$(cat <<EOL
        -e FLAGS_fraction_of_gpu_memory_to_use=0.15 \
        -e CTEST_OUTPUT_ON_FAILURE=1 \
        -e CTEST_PARALLEL_LEVEL=5 \
        -e WITH_GPU=ON \
        -e WITH_TESTING=ON \
        -e WITH_C_API=OFF \
        -e WITH_COVERAGE=ON \
        -e COVERALLS_UPLOAD=ON \
        -e WITH_DEB=OFF \
        -e CMAKE_BUILD_TYPE=RelWithDebInfo \
        -e PADDLE_FRACTION_GPU_MEMORY_TO_USE=0.15 \
        -e CUDA_VISIBLE_DEVICES=0,1 \
        -e WITH_DISTRIBUTE=ON \
        -e RUN_TEST=ON
EOL
    )
    set -x
    nvidia-docker run -it \
        -d \
        --name $CONTAINER_ID \
        ${DOCKER_ENV} \
        -v $PADDLE_ROOT:/paddle \
        -w /paddle \
        $IMG \
        /bin/bash
    set +x
}

function main() {
    DOCKER_REPO="paddlepaddle/paddle"
    VERSION="latest-dev"
    CONTAINER_ID="${USER}_paddle_dev"
    PADDLE_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}")/../../" && pwd )"
    if [ "$1" == "build_android" ]; then
        CONTAINER_ID="${USER}_paddle_dev_android"
        VERSION="latest-dev-android"
    fi
    IMG=${DOCKER_REPO}:${VERSION}

    case $1 in
      start)
        start_build_docker
        ;;
      build_android)
        start_build_docker
        docker exec ${CONTAINER_ID} bash -c "./paddle/scripts/paddle_build.sh $@"
        ;;
      *)
        if container_running "${CONTAINER_ID}"; then
            docker exec ${CONTAINER_ID} bash -c "./paddle/scripts/paddle_build.sh $@"
        else
            echo "Please start container first, with command:"
            echo "$0 start"
        fi
        ;;
    esac
}

main $@
