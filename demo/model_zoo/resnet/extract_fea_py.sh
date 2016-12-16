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

#Note if you use CPU mode, you need to set use_gpu=0 in classify.py. like this:
#conf_args = "is_test=0,use_gpu=1,is_predict=1"
#conf = parse_config(train_conf, conf_args)
#swig_paddle.initPaddle("--use_gpu=0")
python classify.py \
     --job=extract \
     --conf=resnet.py \
     --use_gpu=1 \
     --mean=model/mean_meta_224/mean.meta \
     --model=model/resnet_50 \
     --data=./example/test.list \
     --output_layer="res5_3_branch2c_conv,res5_3_branch2c_bn" \
     --output_dir=features
