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

settings(batch_size=1000, learning_rate=1e-4)

din = data_layer(name='data', size=200)

hidden = fc_layer(input=din, size=200, act=SigmoidActivation())

rnn = recurrent_layer(input=hidden, act=SigmoidActivation())

rnn2 = recurrent_layer(input=hidden, act=SigmoidActivation(), reverse=True)

lstm1_param = fc_layer(
    input=hidden, size=200 * 4, act=LinearActivation(), bias_attr=False)

lstm1 = lstmemory(input=lstm1_param, act=SigmoidActivation())

lstm2_param = fc_layer(
    input=hidden, size=200 * 4, act=LinearActivation(), bias_attr=False)

lstm2 = lstmemory(input=lstm2_param, act=SigmoidActivation(), reverse=True)

gru1_param = fc_layer(
    input=hidden, size=200 * 3, act=LinearActivation(), bias_attr=False)
gru1 = grumemory(input=gru1_param, act=SigmoidActivation())

gru2_param = fc_layer(
    input=hidden, size=200 * 3, act=LinearActivation(), bias_attr=False)
gru2 = grumemory(input=gru2_param, act=SigmoidActivation(), reverse=True)

outputs(
    last_seq(input=rnn),
    first_seq(input=rnn2),
    last_seq(input=lstm1),
    first_seq(input=lstm2),
    last_seq(input=gru1),
    first_seq(gru2))
