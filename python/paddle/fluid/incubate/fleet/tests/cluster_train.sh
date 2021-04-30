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

# start pserver0
python fleet_deep_ctr.py \
    --role pserver \
    --endpoints 127.0.0.1:7000,127.0.0.1:7001 \
    --current_endpoint 127.0.0.1:7000 \
    --trainers 2 \
    > pserver0.log 2>&1 &

# start pserver1
python fleet_deep_ctr.py \
    --role pserver \
    --endpoints 127.0.0.1:7000,127.0.0.1:7001 \
    --current_endpoint 127.0.0.1:7001 \
    --trainers 2 \
    > pserver1.log 2>&1 &

# start trainer0
python fleet_deep_ctr.py \
    --role trainer \
    --endpoints 127.0.0.1:7000,127.0.0.1:7001 \
    --trainers 2 \
    --trainer_id 0 \
    > trainer0.log 2>&1 &

# start trainer1
python fleet_deep_ctr.py \
    --role trainer \
    --endpoints 127.0.0.1:7000,127.0.0.1:7001 \
    --trainers 2 \
    --trainer_id 1 \
    > trainer1.log 2>&1 &
