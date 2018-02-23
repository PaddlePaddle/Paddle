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

:'
Visual deep residual network
1. Using make_model_diagram.py to generate dot file.
2. Using graphviz to convert dot file.

Usage:
./net_diagram.sh
'

set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd $DIR

img_type=png
img_fileprefix=ResNet_50
conf_filename=resnet.py
dot_filename=ResNet_50.dot
config_str="layer_num=50,data_provider=0"

python -m paddle.utils.make_model_diagram $conf_filename $dot_filename $config_str

# If you have installed graphviz, running like this:
# dot -Tpng -o ResNet.png ResNet.dot
