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

settings(batch_size=1000, learning_rate=1e-5)

data = data_layer(name='data', size=2016, height=48, width=42)

conv = img_conv_layer(
    input=data,
    filter_size=3,
    num_channels=1,
    num_filters=16,
    padding=1,
    act=LinearActivation(),
    bias_attr=True)

pool = img_pool_layer(input=conv, pool_size=2, stride=2, pool_type=MaxPooling())

pad = pad_layer(input=pool, pad_c=[2, 3], pad_h=[1, 2], pad_w=[3, 1])

outputs(pad)
