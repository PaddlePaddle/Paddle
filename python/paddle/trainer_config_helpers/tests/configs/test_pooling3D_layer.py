#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.trainer_config_helpers import *

settings(batch_size=100, learning_rate=1e-5)

data_2d = data_layer(name='data_2d', size=6000, height=20, width=10)

pool_2d = img_pool_layer(
    name="pool___2d",
    input=data_2d,
    num_channels=30,
    pool_size=5,
    stride=3,
    padding=1,
    pool_type=AvgPooling())
outputs(pool_2d)

data_3d = data_layer(
    name='data_3d_1', size=60000, depth=10, height=20, width=10)

pool_3d_1 = img_pool3d_layer(
    name="pool_3d_1",
    input=data_3d,
    num_channels=30,
    pool_size=5,
    stride=3,
    padding=1,
    pool_type=AvgPooling())
outputs(pool_3d_1)

pool_3d_2 = img_pool3d_layer(
    name="pool_3d_2",
    input=data_3d,
    num_channels=30,
    pool_size=[5, 5, 5],
    stride=[3, 3, 3],
    padding=[1, 1, 1],
    pool_type=MaxPooling())
outputs(pool_3d_2)
