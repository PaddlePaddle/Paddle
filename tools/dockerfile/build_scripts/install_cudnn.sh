#!/bin/bash

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

# Top-level build script called from Dockerfile

# Stop at any error, show all commands
set -ex

VERSION=$(nvcc --version | grep release | grep -oEi "release ([0-9]+)\.([0-9])"| sed "s/release //")

if [[ "$1" == "cudnn811" && "$VERSION" == "11.1" ]]; then
  wget -q https://paddle-ci.gz.bcebos.com/cudnn/cudnn-11.2-linux-x64-v8.1.1.33.tgz --no-check-certificate
  tar -xzf cudnn-11.2-linux-x64-v8.1.1.33.tgz && \
  cd cuda && \
  cp -r include /usr && \
  cp -r lib64 /usr && cd ../ && \
  rm -f cudnn-11.2-linux-x64-v8.1.1.33.tgz && \
  rm -rf cuda
elif [[ "$1" == "cudnn811" && "$VERSION" == "10.2" ]]; then
  wget -q https://paddle-ci.gz.bcebos.com/cudnn/cudnn-10.2-linux-x64-v8.1.1.33.tgz --no-check-certificate
  tar -xzf cudnn-10.2-linux-x64-v8.1.1.33.tgz && \
  cd cuda && \
  cp -r include /usr && \
  cp -r lib64 /usr && cd ../ && \
  rm -f cudnn-10.2-linux-x64-v8.1.1.33.tgz && \
  rm -rf cuda
elif [[ "$1" == "cudnn821" && "$VERSION" == "11.2" ]]; then
  wget -q https://paddle-ci.gz.bcebos.com/cudnn/cudnn-11.3-linux-x64-v8.2.1.32.tgz --no-check-certificate
  tar -xzf cudnn-11.3-linux-x64-v8.2.1.32.tgz && \
  cd cuda && \
  cp -r include /usr && \
  cp -r lib64 /usr && cd ../ && \
  rm -f cudnn-11.3-linux-x64-v8.2.1.32.tgz && \
  rm -rf cuda
fi
