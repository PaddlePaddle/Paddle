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
paddle train \
  --config=./db_lstm.py \
  --use_gpu=0 \
  --log_period=5000 \
  --trainer_count=1 \
  --show_parameter_stats_period=5000 \
  --save_dir=./output \
  --num_passes=10000 \
  --average_test_period=10000000 \
  --init_model_path=./data \
  --load_missing_parameter_strategy=rand \
  --test_all_data_in_one_period=1 \
  2>&1 | tee 'train.log'
paddle usage -l train.log -e $? -n "semantic_role_labeling_train" >/dev/null 2>&1
