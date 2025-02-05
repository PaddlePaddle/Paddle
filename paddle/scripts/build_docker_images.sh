#!/bin/sh

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

set -xe

REPO="${REPO:-paddlepaddle}"

cp -f ../../python/requirements.txt .

sed 's#FROM nvidia/cuda:8.0-cudnn7-devel-ubuntu16.04#FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04#g' ../../Dockerfile |
sed 's#TensorRT-4.0.1.6-ubuntu14.04.x86_64-gnu.cuda.8.0.cudnn7.0.tar.gz#TensorRT-6.0.1.5.Ubuntu-16.04.x86_64-gnu.cuda-9.0.cudnn7.6.tar.gz#g' |
sed 's#/usr/local/TensorRT#/usr/local/TensorRT-6.0.1.5#g' |
sed 's#libnccl2=2.1.2-1+cuda8.0 libnccl-dev=2.1.2-1+cuda8.0#libnccl2=2.4.7-1+cuda9.0 libnccl-dev=2.4.7-1+cuda9.0#g' |
sed 's#COPY ./paddle/scripts/docker/root/#COPY ./docker/root/#g' |
sed 's#COPY ./python/requirements.txt#COPY ./requirements.txt#' > Dockerfile.cuda9.0-cudnn7
# docker build -t ${REPO}/paddle:cuda9.0-cudnn7-devel-ubuntu16.04 -f Dockerfile.cuda9.0-cudnn7 .

sed 's#FROM nvidia/cuda:8.0-cudnn7-devel-ubuntu16.04#FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04#g' ../../Dockerfile |
sed 's#TensorRT-4.0.1.6-ubuntu14.04.x86_64-gnu.cuda.8.0.cudnn7.0.tar.gz#TensorRT_5.1_ga_cuda10_cudnnv7.5.tar.gz#g' |
sed 's#/usr/local/TensorRT#/usr/local/TensorRT_5.1_ga_cuda10_cudnnv7.5#g' |
sed 's#libnccl2=2.1.2-1+cuda8.0 libnccl-dev=2.1.2-1+cuda8.0#libnccl2=2.4.7-1+cuda10.0 libnccl-dev=2.4.7-1+cuda10.0#g' |
sed 's#COPY ./paddle/scripts/docker/root/#COPY ./docker/root/#g' |
sed 's#COPY ./python/requirements.txt#COPY ./requirements.txt#' > Dockerfile.cuda10.0-cudnn7
# docker build -t ${REPO}/paddle:cuda10.0-cudnn7-devel-ubuntu16.04 -f Dockerfile.cuda10.0-cudnn7 .
