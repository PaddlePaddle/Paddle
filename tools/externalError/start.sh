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
protobuf/bin/protoc -I../../paddle/phi/core/ --python_out . ../../paddle/phi/core/external_error.proto

python3.8 spider.py
tar czvf externalErrorMsg_$(date +'%Y%m%d').tar.gz externalErrorMsg.pb
