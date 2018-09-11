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

import difflib
import unittest

import paddle.trainer_config_helpers as conf_helps
import paddle.v2.activation as activation
import paddle.v2.data_type as data_type
import paddle.v2.layer as layer
from paddle.trainer_config_helpers.config_parser_utils import \
    parse_network_config as parse_network
from paddle.trainer_config_helpers.config_parser_utils import \
    reset_parser


class RNNTest(unittest.TestCase):
    def test_simple_rnn(self):
        dict_dim = 10
        word_dim = 8
        hidden_dim = 8

        def parse_old_rnn():
            reset_parser()

            def step(y):
                mem = conf_helps.memory(name="rnn_state", size=hidden_dim)
                out = conf_helps.fc_layer(
                    input=[y, mem],
                    size=hidden_dim,
                    act=activation.Tanh(),
                    bias_attr=True,
                    name="rnn_state")
                return out

            def test():
                data = conf_helps.data_layer(name="word", size=dict_dim)
                embd = conf_helps.embedding_layer(input=data, size=word_dim)
                conf_helps.recurrent_group(
                    name="rnn", step=step, input=embd, reverse=True)

            return str(parse_network(test))

        def parse_new_rnn():
            reset_parser()

            def new_step(y):
                mem = layer.memory(name="rnn_state", size=hidden_dim)
                out = layer.fc(input=[y, mem],
                               size=hidden_dim,
                               act=activation.Tanh(),
                               bias_attr=True,
                               name="rnn_state")
                return out

            data = layer.data(
                name="word", type=data_type.integer_value(dict_dim))
            embd = layer.embedding(input=data, size=word_dim)
            rnn_layer = layer.recurrent_group(
                name="rnn", step=new_step, input=embd, reverse=True)
            return str(layer.parse_network(rnn_layer))

        diff = difflib.unified_diff(parse_old_rnn().splitlines(1),
                                    parse_new_rnn().splitlines(1))
        print ''.join(diff)

    def test_sequence_rnn_multi_input(self):
        dict_dim = 10
        word_dim = 8
        hidden_dim = 8
        label_dim = 3

        def parse_old_rnn():
            reset_parser()

            def test():
                data = conf_helps.data_layer(name="word", size=dict_dim)
                label = conf_helps.data_layer(name="label", size=label_dim)
                emb = conf_helps.embedding_layer(input=data, size=word_dim)
                boot_layer = conf_helps.data_layer(name="boot", size=10)
                boot_layer = conf_helps.fc_layer(
                    name='boot_fc', input=boot_layer, size=10)

                def step(y, wid):
                    z = conf_helps.embedding_layer(input=wid, size=word_dim)
                    mem = conf_helps.memory(
                        name="rnn_state",
                        size=hidden_dim,
                        boot_layer=boot_layer)
                    out = conf_helps.fc_layer(
                        input=[y, z, mem],
                        size=hidden_dim,
                        act=conf_helps.TanhActivation(),
                        bias_attr=True,
                        name="rnn_state")
                    return out

                out = conf_helps.recurrent_group(
                    name="rnn", step=step, input=[emb, data])

                rep = conf_helps.last_seq(input=out)
                prob = conf_helps.fc_layer(
                    size=label_dim,
                    input=rep,
                    act=conf_helps.SoftmaxActivation(),
                    bias_attr=True)

                conf_helps.outputs(
                    conf_helps.classification_cost(
                        input=prob, label=label))

            return str(parse_network(test))

        def parse_new_rnn():
            reset_parser()
            data = layer.data(
                name="word", type=data_type.dense_vector(dict_dim))
            label = layer.data(
                name="label", type=data_type.dense_vector(label_dim))
            emb = layer.embedding(input=data, size=word_dim)
            boot_layer = layer.data(
                name="boot", type=data_type.dense_vector(10))
            boot_layer = layer.fc(name='boot_fc', input=boot_layer, size=10)

            def step(y, wid):
                z = layer.embedding(input=wid, size=word_dim)
                mem = layer.memory(
                    name="rnn_state", size=hidden_dim, boot_layer=boot_layer)
                out = layer.fc(input=[y, z, mem],
                               size=hidden_dim,
                               act=activation.Tanh(),
                               bias_attr=True,
                               name="rnn_state")
                return out

            out = layer.recurrent_group(
                name="rnn", step=step, input=[emb, data])

            rep = layer.last_seq(input=out)
            prob = layer.fc(size=label_dim,
                            input=rep,
                            act=activation.Softmax(),
                            bias_attr=True)

            cost = layer.classification_cost(input=prob, label=label)

            return str(layer.parse_network(cost))

        diff = difflib.unified_diff(parse_old_rnn().splitlines(1),
                                    parse_new_rnn().splitlines(1))
        print ''.join(diff)


if __name__ == '__main__':
    unittest.main()
