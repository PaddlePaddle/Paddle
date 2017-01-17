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
cd ..

paddle train \
    --config='paraphrase/train.conf' \
    --save_dir='paraphrase/model' \
    --init_model_path='data/paraphrase_model' \
    --load_missing_parameter_strategy=rand \
    --use_gpu=false \
    --num_passes=16 \
    --show_parameter_stats_period=100 \
    --trainer_count=4 \
    --log_period=10 \
    --dot_period=5 \
    2>&1 | tee 'paraphrase/train.log'
paddle usage -l 'paraphrase/train.log' -e $? -n "seqToseq_paraphrase_train" >/dev/null 2>&1
