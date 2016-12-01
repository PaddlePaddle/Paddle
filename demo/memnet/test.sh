#!/bin/bash
# Copyright (c) 2016 Baidu, Inc. All Rights Reserved
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
config=./external_memory_example.conf
log=test.log

# change the following path to the model to be tested
evaluate_pass="./mem_model/pass-00030"

echo 'evaluating from pass '$evaluate_pass
model_list=./model.list
touch $model_list | echo $evaluate_pass > $model_list

paddle train \
  -v=5 \
  --config=$config \
  --model_list=$model_list \
  --job=test \
  --use_gpu=1 \
  --config_args=is_test=1 \
  2>&1 | tee $log
