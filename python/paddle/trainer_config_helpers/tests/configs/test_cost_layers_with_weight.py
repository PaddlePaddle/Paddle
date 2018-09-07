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

settings(learning_rate=1e-4, batch_size=1000)

data = data_layer(name='input', size=300)
lbl = data_layer(name='label', size=1)
wt = data_layer(name='weight', size=1)
fc = fc_layer(input=data, size=10, act=SoftmaxActivation())

outputs(
    classification_cost(
        input=fc, label=lbl, weight=wt),
    square_error_cost(
        input=fc, label=lbl, weight=wt),
    nce_layer(
        input=fc,
        label=data_layer(
            name='multi_class_label', size=500),
        weight=wt))
