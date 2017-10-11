#!/bin/bash
# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
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

# Should run pserver.sh before run this script.
bin_dir=$(cd `dirname $0`; pwd)
home_dir=$(cd "${bin_dir}/.."; pwd)
source "$bin_dir/env.sh"

model_dir="$bin_dir/output"
log_file="$bin_dir/train.log"

pushd "$home_dir"
cfg=trainer_config.lr.py
paddle train \
  --start_pserver=false \
  --config=$cfg \
  --save_dir=${model_dir} \
  --trainer_count=4 \
  --local=0 \
  --log_period=100 \
  --num_passes=15 \
  --use_gpu=false \
  --show_parameter_stats_period=100 \
  --test_all_data_in_one_period=1 \
  --num_gradient_servers=1 \
  --nics=`get_nics` \
  --port=7164 \
  --ports_num=1 \
  --pservers="127.0.0.1" \
  --comment="paddle_trainer" \
  2>&1 | tee "$log_file"
popd
