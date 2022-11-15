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

temp_dir=$(mktemp --directory)
pushd ${temp_dir} \
&& git clone ${PLUGIN_URL} \
&& pushd PaddleCustomDevice/ \
&& git fetch origin \
&& git checkout ${PLUGIN_TAG} -b dev \ 
&& pushd backends/custom_cpu \
&& mkdir build && pushd build && cmake .. && make -j8 && popd && popd && popd && popd

echo "begin test use custom_cpu"

export FLAGS_selected_custom_cpus=0,1
export CUSTOM_CPU_VISIBLE_DEVICES=0,1
export CUSTOM_DEVICE_ROOT=${temp_dir}/PaddleCustomDevice/backends/custom_cpu/build

distributed_args="--devices=0,1"
python -m paddle.distributed.fleet.launch ${distributed_args} custom_device_multi_process_collective.py fleetlaunch_custom_cpu
