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

seq = data_layer(name='seq_input', size=100)
sub_seq = data_layer(name='sub_seq_input', size=100)
lbl = data_layer(name='label', size=1)


def generate_rnn_simple(name):
    def rnn_simple(s):
        m = memory(name=name, size=200)
        fc = fc_layer(input=[s, m], size=200, name=name)
        return fc

    return rnn_simple


def generate_rnn_simple_no_name():
    def rnn_simple(s):
        m = memory(name=None, size=200)
        fc = fc_layer(input=[s, m], size=200)
        m.set_input(fc)
        return fc

    return rnn_simple


with mixed_layer() as lstm_param:  # test lstm unit, rnn group
    lstm_param += full_matrix_projection(input=seq, size=100 * 4)

with mixed_layer() as gru_param:
    gru_param += full_matrix_projection(input=seq, size=100 * 3)

outputs(
    last_seq(input=recurrent_group(
        step=generate_rnn_simple('rnn_forward'), input=seq)),
    first_seq(input=recurrent_group(
        step=generate_rnn_simple('rnn_back'), input=seq, reverse=True)),
    last_seq(input=recurrent_group(
        step=generate_rnn_simple('rnn_subseq_forward'),
        input=SubsequenceInput(input=sub_seq))),
    last_seq(input=lstmemory_group(
        input=lstm_param, size=100)),
    last_seq(input=gru_group(
        input=gru_param, size=100)),
    last_seq(input=recurrent_group(
        step=generate_rnn_simple_no_name(), input=seq)), )
