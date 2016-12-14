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

# Download the preprocessed data
wget http://paddlepaddle.bj.bcebos.com/demo/quick_start_preprocessed_data/preprocessed_data.tar.gz

# Extract package
tar zxvf preprocessed_data.tar.gz

# Remove compressed package
rm preprocessed_data.tar.gz
