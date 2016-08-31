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

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd $DIR

mkdir model
cd model

echo "Downloading ResNet models..."

for file in resnet_50.tar.gz resnet_101.tar.gz resnet_152.tar.gz mean_meta_224.tar.gz 
do 
  # following is the google drive address
  # you can also directly download from https://pan.baidu.com/s/1o8q577s
  wget https://www.googledrive.com/host/0B7Q8d52jqeI9ejh6Q1RpMTFQT1k/imagenet/$file --no-check-certificate
  tar -xvf $file 
  rm $file
done

echo "Done."
