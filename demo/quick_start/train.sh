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

cfg=trainer_config.lr.py
#cfg=trainer_config.emb.py
#cfg=trainer_config.cnn.py
#cfg=trainer_config.lstm.py
#cfg=trainer_config.bidi-lstm.py
#cfg=trainer_config.db-lstm.py
#cfg=trainer_config.resnet-lstm.py
paddle train \
  --config=$cfg \
  --save_dir=./output \
  --trainer_count=4 \
  --log_period=100 \
  --num_passes=15 \
  --use_gpu=false \
  --show_parameter_stats_period=100 \
  --test_all_data_in_one_period=1 \
  2>&1 | tee 'train.log'
