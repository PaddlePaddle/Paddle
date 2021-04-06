#!/bin/bash

# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

PADDLE_ROOT=/home
mkdir ${PADDLE_ROOT}
cd ${PADDLE_ROOT}
pip install /paddle/build/opt/paddle/share/wheels/*.whl
git clone https://github.com/PaddlePaddle/FluidDoc
git clone https://github.com/tianshuo78520a/PaddlePaddle.org.git
cd  ${PADDLE_ROOT}/PaddlePaddle.org
git reset 3feaa68376d8423e41d076814e901e6bf108c705
cd ${PADDLE_ROOT}/FluidDoc/doc/fluid/api
sh gen_doc.sh
apt-get update && apt-get install -y python-dev build-essential
cd ${PADDLE_ROOT}/PaddlePaddle.org/portal
pip install -r requirements.txt
#If the default port is not occupied, you can use port 8000, you need to replace it with a random port on the CI.
sed -i "s#8000#$1#g" runserver
nohup ./runserver --paddle ${PADDLE_ROOT}/FluidDoc &
