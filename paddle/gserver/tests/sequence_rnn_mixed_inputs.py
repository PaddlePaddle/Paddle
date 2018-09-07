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
    obj='process_mixed')

settings(batch_size=2, learning_rate=0.01)
######################## network configure ################################
dict_dim = 10
word_dim = 2
hidden_dim = 2
label_dim = 2

data1 = data_layer(name="word1", size=dict_dim)
data2 = data_layer(name="word2", size=dict_dim)
label = data_layer(name="label", size=label_dim)

encoding = embedding_layer(input=data2, size=word_dim)


# This hierarchical RNN is designed to be equivalent to the simple RNN in
# sequence_rnn_matched_inputs.conf
def outer_step(subseq, seq, nonseq, encoding):
    outer_mem = memory(name="outer_rnn_state", size=hidden_dim)

    def inner_step(data1, data2, label):
        inner_mem = memory(
            name="inner_rnn_state", size=hidden_dim, boot_layer=outer_mem)

        subseq = embedding_layer(input=data1, size=word_dim)
        seq = embedding_layer(input=data2, size=word_dim)
        nonseq = embedding_layer(input=label, size=word_dim)

        print_layer(input=[data1, seq, label, inner_mem])
        out = fc_layer(
            input=[subseq, seq, nonseq, inner_mem],
            size=hidden_dim,
            act=TanhActivation(),
            bias_attr=True,
            name='inner_rnn_state')
        return out

    decoder = recurrent_group(
        step=inner_step, name='inner',
        input=[subseq, StaticInput(seq), nonseq])
    last = last_seq(name="outer_rnn_state", input=decoder)
    context = simple_attention(
        encoded_sequence=encoding, encoded_proj=encoding, decoder_state=last)
    return context


out = recurrent_group(
    name="outer",
    step=outer_step,
    input=[data1, data2, StaticInput(label), StaticInput(encoding)])

rep = last_seq(input=out)
prob = fc_layer(
    size=label_dim, input=rep, act=SoftmaxActivation(), bias_attr=True)

outputs(classification_cost(input=prob, label=label))
