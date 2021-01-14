#!/usr/bin/env bash
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
