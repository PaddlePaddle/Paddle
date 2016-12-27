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

#Note the default model is pass-00002, you shold make sure the model path
#exists or change the mode path.
#only test on trainer_config.lr.py
model=output/model/pass-00001/
config=trainer_config.lr.py
label=data/labels.list
dict=data/dict.txt
batch_size=20
head -n$batch_size data/test.txt | python api_predict.py \
     --tconf=$config\
     --model=$model \
     --label=$label \
     --dict=$dict \
     --batch_size=$batch_size
