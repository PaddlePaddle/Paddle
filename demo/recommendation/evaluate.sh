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
  cat $1  | grep -Pzo 'Test .*\n.*pass-.*' | sed  -r 'N;s/Test.* cost=([0-9]+\.[0-9]+).*\n.*pass-([0-9]+)/\1 \2/g' | sort | head -n 1
}

LOG=`get_best_pass log.txt`
LOG=(${LOG})
echo 'Best pass is '${LOG[1]}, ' error is '${LOG[0]}, 'which means predict get error as '`echo ${LOG[0]} | python -c 'import math; print math.sqrt(float(raw_input()))/2'`

evaluate_pass="output/pass-${LOG[1]}"

echo 'evaluating from pass '$evaluate_pass
