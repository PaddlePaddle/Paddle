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
config=vgg_16_mnist.py
output=./mnist_vgg_model
log=train.log

paddle train \
--config=$config \
--dot_period=10 \
--log_period=100 \
--test_all_data_in_one_period=1 \
--use_gpu=0 \
--trainer_count=1 \
--num_passes=100 \
--save_dir=$output \
2>&1 | tee $log
paddle usage -l $log -e $? -n "mnist_train" >/dev/null 2>&1

python -m paddle.utils.plotcurve -i $log > plot.png
