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

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd $DIR

mkdir model
cd model

echo "Downloading ResNet models..."

for file in resnet_50.tar.gz resnet_101.tar.gz resnet_152.tar.gz mean_meta_224.tar.gz 
do 
  wget http://paddlepaddle.bj.bcebos.com/model_zoo/imagenet/$file
  tar -xvf $file 
  rm $file
done

echo "Done."
