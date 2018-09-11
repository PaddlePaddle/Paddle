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

data_1 = data_layer(name='data_a', size=100)
data_2 = data_layer(name='data_b', size=100)

mixed_param = ParamAttr(name='mixed_param')

with mixed_layer(size=400, bias_attr=False) as m1:
    m1 += full_matrix_projection(input=data_1, param_attr=mixed_param)

with mixed_layer(size=400, bias_attr=False) as m2:
    m2 += full_matrix_projection(input=data_2, param_attr=mixed_param)

lstm_param = ParamAttr(name='lstm_param')
lstm_bias = ParamAttr(name='lstm_bias', initial_mean=0., initial_std=0.)

lstm1 = lstmemory_group(
    input=m1,
    param_attr=lstm_param,
    lstm_bias_attr=lstm_bias,
    input_proj_bias_attr=False)

lstm2 = lstmemory_group(
    input=m2,
    param_attr=lstm_param,
    lstm_bias_attr=lstm_bias,
    input_proj_bias_attr=False)

softmax_param = ParamAttr(name='softmax_param')

predict = fc_layer(
    input=[last_seq(input=lstm1), last_seq(input=lstm2)],
    size=10,
    param_attr=[softmax_param, softmax_param],
    bias_attr=False,
    act=SoftmaxActivation())
outputs(
    classification_cost(
        input=predict, label=data_layer(
            name='label', size=10)))
