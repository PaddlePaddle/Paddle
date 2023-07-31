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

BUILD_FROM_SCRATCH=0
MAX_JOBS=8
BUILD_ARTIFACTS_URL="https://oss.mthreads.com/mt-ai-data/paddle_musa/pre_build/paddle_musa_build_artifacts.tar.gz"
CCACHE_ARTIFACTS_URL="https://oss.mthreads.com/mt-ai-data/paddle_musa/pre_build/paddle_musa_ccache_artifacts.tar.gz"

CUR_DIR=$(cd "$(dirname "$0")"; pwd)
PADDLE_MUSA_DIR=$(dirname "$CUR_DIR")

usage() {
  echo -e "\033[1;32mThis script is used to build paddle_musa on CI. \033[0m"
  echo -e "\033[1;32mParameters usage: \033[0m"
  echo -e "\033[32m    --from_scratch    : Means building paddle_musa from scratch. \033[0m"
  echo -e "\033[32m    -j/--jobs         : Means number of threads used for compiling. \033[0m"
  echo -e "\033[32m    -h/--help         : Help information. \033[0m"
}

parameters=`getopt -o j:h:: --long jobs:,from_scratch,help::, -n "$0" -- "$@"`
[ $? -ne 0 ] && { echo -e "\033[34mTry '$0 --help' for more information. \033[0m"; exit 1; }

eval set -- "$parameters"

while true;do
    case "$1" in
        --from_scratch) BUILD_FROM_SCRATCH=1; shift ;;
        -j|--jobs) MAX_JOBS=$2 ; shift 2;;
        -h|--help) usage;exit ;;
        --)
            shift ; break ;;
        *) usage;exit 1;;
    esac
done

pushd ${PADDLE_MUSA_DIR}
# prepare submodules by copying from the local repo,
# in this case, CI docker need to be updated once submodules' version changed
cp -r ${PADDLE_MUSA_REPO_PATH}/third_party/. third_party
cp -r ${PADDLE_MUSA_REPO_PATH}/.git/modules .git

export INFERENCE_DEMO_INSTALL_DIR="/home/data/paddle_musa/.cache/build"
export CCACHE_DIR="/home/data/paddle_musa/.ccache/build"


CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "${CURRENT_BRANCH}" = "develop" ]; then
  ref_revision=$(git log -2 --pretty=%H | tail -1)
else
  ref_revision="origin/develop"
fi

cmake_diff_nums=$(git diff --name-only ${ref_revision} | grep "CMake\|cmake" | wc -l)
if [ ${cmake_diff_nums} -gt 0 ]; then
  BUILD_FROM_SCRATCH=1
fi

ccache -s

if [ ${BUILD_FROM_SCRATCH} -eq 0 ]; then
  # prepare the pre-built artifacts
  wget --no-check-certificate ${CCACHE_ARTIFACTS_URL} -O ./ccache_artifacts.tar.gz
  wget --no-check-certificate ${BUILD_ARTIFACTS_URL} -O ./build_artifacts.tar.gz

  tar zmxf build_artifacts.tar.gz
  tar zmxf ccache_artifacts.tar.gz -C ${CCACHE_DIR}
  rm ccache_artifacts.tar.gz build_artifacts.tar.gz
  rm build/build_summary.txt
fi

git diff --name-only ${ref_revision} | xargs touch

find build -name "CMakeCache*" | xargs rm
WITH_MKL=ON WITH_AVX=ON PY_VERSION=3.8 /bin/bash paddle/scripts/paddle_build.sh build_only ${MAX_JOBS}
find ./dist -name *whl | xargs pip install

popd
