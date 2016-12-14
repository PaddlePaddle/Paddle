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

#set names of layer which you want to extract feature
#in Outputs() of resnet.py 
#like: Outputs("res5_3_branch2c_conv", "res5_3_branch2c_bn")
layer_num=50
configure=./resnet.py
model_path=./model/resnet_$layer_num
fea_dir=fea_output
#Output is text file.
#Each line is one sample's features.
#If you set N layer names in Outputs()
#each line contains N features sperated by ";". 

# create model list file.
model_list=./model.list
touch $model_list | echo $model_path > $model_list

paddle train \
  --local=true \
  --job=test \
  --config=$configure \
  --model_list=$model_list \
  --use_gpu=1 \
  --predict_output_dir=$fea_dir \
  --config_args=is_test=1,layer_num=$layer_num
