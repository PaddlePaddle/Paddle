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
set -x

# download the pretrained model
# following is the google drive address
# you can also directly download from https://pan.baidu.com/s/1o8q577s
wget https://www.googledrive.com/host/0B7Q8d52jqeI9ejh6Q1RpMTFQT1k/wmt14_model.tar.gz --no-check-certificate

# untar the model
tar -zxvf wmt14_model.tar.gz
rm wmt14_model.tar.gz 
