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

weight = data_layer(name='w', size=1)
a = data_layer(name='a', size=100)
b = data_layer(name='b', size=100)
c = data_layer(name='c', size=200)
d = data_layer(name='d', size=31)

outputs(
    interpolation_layer(
        input=[a, b], weight=weight),
    power_layer(
        input=a, weight=weight),
    scaling_layer(
        input=a, weight=weight),
    cos_sim(
        a=a, b=b),
    cos_sim(
        a=a, b=c, size=2),
    sum_to_one_norm_layer(input=a),
    conv_shift_layer(
        a=a, b=d),
    tensor_layer(
        a=a, b=b, size=1000),
    slope_intercept_layer(
        input=a, slope=0.7, intercept=0.9),
    linear_comb_layer(
        weights=b, vectors=c))
