#!/usr/bin/env bash

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

#=================================================
#                   Paddle Plugin
#=================================================

PLUGIN_ROOT=${PADDLE_ROOT}/plugin

# path 
rm -rf ${PLUGIN_ROOT} && mkdir -p ${PLUGIN_ROOT} && cd ${PLUGIN_ROOT} &&

# Ascend910
# source
PLUGIN_NPU_ROOT=${PLUGIN_ROOT}/paddlepaddle_ascend910
git clone -b master https://github.com/ronny1996/paddlepaddle_ascend910.git &&

# build
cd ${PLUGIN_NPU_ROOT} &&
mkdir build && cd build &&
cmake .. -DWITH_KERNELS=ON -DPADDLE_ROOT=${PADDLE_ROOT} && make &&

# install
pip uninstall -y paddle_ascend910 &&
pip install dist/*.whl &&

# run test
cd ${PLUGIN_NPU_ROOT}/tests &&
python ./test_MNIST_model.py;error_code=$?


if [ "$error_code" != 0 ];then
    echo "Check Plugin failed!"
    exit 8;
fi
