#!/bin/bash

# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

set -e

IMAGE_DOCKER_NAME=NULL
TAG=latest
MUSA_TOOLKITS_URL=""
MUDNN_URL=""
DOCKER_FILE="./Dockerfile.musa"

usage() {
  echo -e "\033[1;32mThis script is used to build docker image for paddle_musa. \033[0m"
  echo -e "\033[1;32mParameters usage: \033[0m"
  echo -e "\033[32m    -i/--image_docker_name     : Name of the docker image. \033[0m"
  echo -e "\033[32m    -t/--tag                   : Tag of the docker image. \033[0m"
  echo -e "\033[32m    -m/--musa_toolkits_url     : The download link of MUSA ToolKit. \033[0m"
  echo -e "\033[32m    -n/--mudnn_url             : The download link of MUDNN. \033[0m"
  echo -e "\033[32m    -f/--file                  : The dockerfile used for docker image building. \033[0m"
  echo -e "\033[32m    -h/--help                  : Help information. \033[0m"
}

# parse parameters
parameters=$(getopt -o i:t:m:n:f:h:: --long image_docker_name:,tag:,musa_toolkits_url:,mudnn_url:,file:,help::, -n "$0" -- "$@")
[ $? -ne 0 ] && { echo -e "\033[34mTry '$0 --help' for more information. \033[0m"; exit 1; }

eval set -- "$parameters"

while true;do
  case "$1" in
    -i|--image_docker_name) IMAGE_DOCKER_NAME=$2; shift 2;;
    -t|--tag) TAG=$2; shift 2;;
    -m|--musa_toolkits_url) MUSA_TOOLKITS_URL=$2; shift 2;;
    -n|--mudnn_url) MUDNN_URL=$2; shift 2;;
    -f|--file) DOCKER_FILE=$2; shift 2;;
    -h|--help) usage; exit ;;
    --) shift ; break ;;
    *) usage; exit 1 ;;
  esac
done

DOCKER_DIR=$(cd "$(dirname "$0")"; pwd)

pushd ${DOCKER_DIR}
build_docker_cmd_prefix="docker build --no-cache --network=host "
build_docker_cmd=${build_docker_cmd_prefix}"--build-arg MUSA_TOOLKITS_URL=${MUSA_TOOLKITS_URL}               \
                                            --build-arg MUDNN_URL=${MUDNN_URL}                               \
                                            -t ${IMAGE_DOCKER_NAME}:${TAG}                                   \
                                            -f ${DOCKER_FILE} ."
eval $build_docker_cmd
popd
