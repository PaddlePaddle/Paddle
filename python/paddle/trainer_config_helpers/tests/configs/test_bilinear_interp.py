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

data = data_layer(name='data', size=2304)

conv = img_conv_layer(
    input=data,
    filter_size=3,
    num_channels=1,
    num_filters=16,
    padding=1,
    act=LinearActivation(),
    bias_attr=True)

bilinear = bilinear_interp_layer(input=conv, out_size_x=64, out_size_y=64)

pool = img_pool_layer(
    input=bilinear,
    num_channels=16,
    pool_size=2,
    stride=2,
    pool_type=MaxPooling())

fc = fc_layer(input=pool, size=384, bias_attr=False)

outputs(fc)
