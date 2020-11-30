#!/bin/bash

# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

set -x

PADDLE_ROOT=$1
TURN_ON_MKL=$2 # use MKL or Openblas

# download models
function download() {
    wget -q http://paddle-tar.bj.bcebos.com/train_demo/LR-1-7/main_program
    wget -q http://paddle-tar.bj.bcebos.com/train_demo/LR-1-7/startup_program
}

download

# build demo trainer
paddle_install_dir=${PADDLE_ROOT}/build/paddle_install_dir

mkdir -p build
cd build
rm -rf *
cmake .. -DPADDLE_LIB=$paddle_install_dir \
         -DWITH_MKLDNN=$TURN_ON_MKL \
         -DWITH_MKL=$TURN_ON_MKL
make

cd ..

# run demo trainer
build/demo_trainer
