#!/usr/bin/env bash

# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

set -ex
SYSTEM=`uname -s`
rm -f protoc-3.11.3-linux-x86_64.*
if [ "$SYSTEM" == "Linux" ]; then
    wget --no-check-certificate https://github.com/protocolbuffers/protobuf/releases/download/v3.11.3/protoc-3.11.3-linux-x86_64.zip
    unzip -d protobuf -o protoc-3.11.3-linux-x86_64.zip
    rm protoc-3.11.3-linux-x86_64.*
elif [ "$SYSTEM" == "Darwin" ]; then
    wget --no-check-certificate https://github.com/protocolbuffers/protobuf/releases/download/v3.11.3/protoc-3.11.3-osx-x86_64.zip
    unzip -d protobuf -o protoc-3.11.3-osx-x86_64.zip
    rm protoc-3.11.3-osx-x86_64.*
else
    echo "please run on Mac/Linux"
    exit 1
fi
protobuf/bin/protoc -I../../paddle/fluid/platform/ --python_out . ../../paddle/fluid/platform/cuda_error.proto

version=90,100,-1    # -1 represent the latest cuda-version 
url=https://docs.nvidia.com/cuda/archive/9.0/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g3f51e3575c2178246db0a94a430e0038,https://docs.nvidia.com/cuda/archive/10.0/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g3f51e3575c2178246db0a94a430e0038,https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g3f51e3575c2178246db0a94a430e0038

if [ "$1" != "" ]; then
    version=$version,$(($1*10))
    if [ "$2" != "" ]; then
        url=$url,$2
    else
        url=$url,https://docs.nvidia.com/cuda/archive/$1/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g3f51e3575c2178246db0a94a430e0038
    fi
fi

python spider.py --version=$version --url=$url
tar czf cudaErrorMessage.tar.gz cudaErrorMessage.pb
