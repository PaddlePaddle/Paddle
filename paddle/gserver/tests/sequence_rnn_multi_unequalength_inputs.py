#  Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

from paddle.trainer_config_helpers import *

######################## data source ################################
define_py_data_sources2(
    train_list='gserver/tests/Sequence/dummy.list',
    test_list=None,
    module='rnn_data_provider',
    obj='process_unequalength_seq')

settings(batch_size=2, learning_rate=0.01)
######################## network configure ################################
dict_dim = 10
word_dim = 8
hidden_dim = 8
label_dim = 2

speaker1 = data_layer(name="word1", size=dict_dim)
speaker2 = data_layer(name="word2", size=dict_dim)

emb1 = embedding_layer(input=speaker1, size=word_dim)
emb2 = embedding_layer(input=speaker2, size=word_dim)

# This hierachical RNN is designed to be equivalent to the RNN in
# sequence_nest_rnn_multi_unequalength_inputs.conf


def step(x1, x2):
    def calrnn(y):
        mem = memory(name='rnn_state_' + y.name, size=hidden_dim)
        out = fc_layer(
            input=[y, mem],
            size=hidden_dim,
            act=TanhActivation(),
            bias_attr=True,
            name='rnn_state_' + y.name)
        return out

    encoder1 = calrnn(x1)
    encoder2 = calrnn(x2)
    return [encoder1, encoder2]


encoder1_rep, encoder2_rep = recurrent_group(
    name="stepout", step=step, input=[emb1, emb2])

encoder1_last = last_seq(input=encoder1_rep)
encoder1_expandlast = expand_layer(input=encoder1_last, expand_as=encoder2_rep)
context = mixed_layer(
    input=[
        identity_projection(encoder1_expandlast),
        identity_projection(encoder2_rep)
    ],
    size=hidden_dim)

rep = last_seq(input=context)
prob = fc_layer(
    size=label_dim, input=rep, act=SoftmaxActivation(), bias_attr=True)

outputs(
    classification_cost(
        input=prob, label=data_layer(
            name="label", size=label_dim)))
