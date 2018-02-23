#!/bin/bash
# Copyright (c) 2016 PaddlePaddle Authors, Inc. All Rights Reserved
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

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd $DIR

#download the dataset
echo "Downloading traffic data..."
wget http://paddlepaddle.cdn.bcebos.com/demo/traffic/traffic_data.tar.gz

#extract package
echo "Unzipping..."
tar -zxvf traffic_data.tar.gz

echo "data/speeds.csv" > train.list
echo "data/speeds.csv" > test.list
echo "data/speeds.csv" > pred.list

echo "Done."
