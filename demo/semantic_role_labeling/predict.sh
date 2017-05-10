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
  sed  -r 'N;s/Test.* cost=([0-9]+\.[0-9]+).*\n.*pass-([0-9]+)/\1 \2/g' | \
  sort -n | head -n 1
}   

log=train.log
LOG=`get_best_pass $log`
LOG=(${LOG})
best_model_path="output/pass-${LOG[1]}"

config_file=db_lstm.py
dict_file=./data/wordDict.txt
label_file=./data/targetDict.txt 
predicate_dict_file=./data/verbDict.txt
input_file=./data/feature
output_file=predict.res
 
python predict.py \
     -c $config_file \
     -w $best_model_path \
     -l $label_file \
     -p $predicate_dict_file  \
     -d $dict_file \
     -i $input_file \
     -o $output_file
