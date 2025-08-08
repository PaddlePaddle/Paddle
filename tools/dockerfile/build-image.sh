#!/bin/bash

# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -ex

# Usage:
# export CANN_VERSION=8.0.RC1
# bash build-image.sh ${CANN_VERSION}

CANN_VERSION=${1:-8.0.0} # default 8.0.0
SYSTEM=${2:-x86_64} # default x86_64
NPU_VERSION=${3:-910b} # default 910b

DOCKER_VERSION=${CANN_VERSION//[^0-9A-Z]/} # 80T13

# Download packages from https://www.hiascend.com/software/cann/community first
if [ ! -f Ascend-cann-toolkit_${CANN_VERSION}_linux-$(uname -m).run ]; then
  echo "Please download CANN installation packages first!"
  exit 1
fi

sed "s#<baseimg>#registry.baidubce.com/device/paddle-cpu:ubuntu20-npu-base-$(uname -m)-gcc84#g" Dockerfile.npu.ubuntu20.gcc84 > Dockerfile.npu.ubuntu20.gcc84.test
docker build --network=host -f Dockerfile.npu.ubuntu20.gcc84.test \
  --build-arg CANN_VERSION=${CANN_VERSION} \
  --build-arg SYSTEM=${SYSTEM} \
  --build-arg NPU_VERSION=${NPU_VERSION} \
  --build-arg http_proxy=${proxy} \
  --build-arg https_proxy=${proxy} \
  --build-arg ftp_proxy=${proxy} \
  --build-arg no_proxy=bcebos.com \
  -t ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-npu:cann${DOCKER_VERSION}-ubuntu20-npu-${NPU_VERSION}-base-$(uname -m)-gcc84 .
docker push ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-npu:cann${DOCKER_VERSION}-ubuntu20-npu-${NPU_VERSION}-base-$(uname -m)-gcc84
rm -rf Dockerfile.npu.ubuntu20.gcc84.test
