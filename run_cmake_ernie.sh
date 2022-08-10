#!/bin/bash -ex

# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

cmake -G "Unix Makefiles" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=YES \
    -DWITH_CONTRIB=OFF \
    -DWITH_DISTRIBUTE=OFF \
    -DWITH_DGC=OFF \
    -DWITH_MKL=ON \
    -DWITH_GRPC=OFF \
    -DWITH_MKLDNN=OFF \
    -DWITH_AVX=OFF \
    -DWITH_TESTING=ON \
    -DWITH_INFERENCE_API_TEST=ON \
    -DWITH_PYTHON=ON \
    -DON_INFER=ON \
    -DWITH_CINN=ON \
    -DTENSORRT_ROOT=/work/lib/TensorRT-8.0.3.4/ \
    -DCUDNN_ROOT=/work/lib/cudnn_v8.2.1/cuda/ \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -B build/ .

