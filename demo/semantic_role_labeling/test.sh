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

function get_best_pass() {
  cat $1  | grep -Pzo 'Test .*\n.*pass-.*' | \
  sed  -r 'N;s/Test.* cost=([0-9]+\.[0-9]+).*\n.*pass-([0-9]+)/\1 \2/g' |\
  sort -n | head -n 1
}

log=train.log
LOG=`get_best_pass $log`
LOG=(${LOG})
evaluate_pass="output/pass-${LOG[1]}"

echo 'evaluating from pass '$evaluate_pass
model_list=./model.list
touch $model_list | echo $evaluate_pass > $model_list

paddle train \
  --config=./db_lstm.py \
  --model_list=$model_list \
  --job=test \
  --use_gpu=false \
  --config_args=is_test=1 \
  --test_all_data_in_one_period=1 \
2>&1 | tee 'test.log'
paddle usage -l test.log -e $? -n "semantic_role_labeling_test" >/dev/null 2>&1
