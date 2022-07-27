#!/bin/bash

# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

rm -rf PaddleCustomDevice && git clone https://github.com/PaddlePaddle/PaddleCustomDevice.git && pushd PaddleCustomDevice/backends/custom_cpu && mkdir build && pushd build && cmake .. && make -j8 && popd && popd

echo "begin test use custom_cpu"

export FLAGS_selected_custom_cpus=0,1
export CUSTOM_DEVICE_ROOT=PaddleCustomDevice/backends/custom_cpu/build

python test_collective_process_group_xccl.py
