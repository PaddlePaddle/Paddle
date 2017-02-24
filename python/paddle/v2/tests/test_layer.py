# Copyright PaddlePaddle contributors. All Rights Reserved
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
import paddle.v2.attr as attr
import paddle.v2.data_type as data_type
import paddle.v2.layer as layer
from paddle.trainer_config_helpers.config_parser_utils import \
    parse_network_config as parse_network

pixel = layer.data(name='pixel', type=data_type.dense_vector(784))
label = layer.data(name='label', type=data_type.integer_value(10))
weight = layer.data(name='weight', type=data_type.dense_vector(10))
score = layer.data(name='score', type=data_type.dense_vector(1))
hidden = layer.fc(input=pixel,
                  size=100,
                  act=activation.Sigmoid(),
                  param_attr=attr.Param(name='hidden'))
inference = layer.fc(input=hidden, size=10, act=activation.Softmax())


class CostLayerTest(unittest.TestCase):
    def test_cost_layer(self):
        cost1 = layer.classification_cost(input=inference, label=label)
        cost2 = layer.classification_cost(
            input=inference, label=label, weight=weight)
        cost3 = layer.cross_entropy_cost(input=inference, label=label)
        cost4 = layer.cross_entropy_with_selfnorm_cost(
            input=inference, label=label)
        cost5 = layer.regression_cost(input=inference, label=label)
        cost6 = layer.regression_cost(
            input=inference, label=label, weight=weight)
        cost7 = layer.multi_binary_label_cross_entropy_cost(
            input=inference, label=label)
        cost8 = layer.rank_cost(left=score, right=score, label=score)
        cost9 = layer.lambda_cost(input=inference, score=score)
        cost10 = layer.sum_cost(input=inference)
        cost11 = layer.huber_cost(input=score, label=label)

        print layer.parse_network(cost1, cost2)
        print layer.parse_network(cost3, cost4)
        print layer.parse_network(cost5, cost6)
        print layer.parse_network(cost7, cost8, cost9, cost10, cost11)


class RNNTest(unittest.TestCase):
    def test_simple_rnn(self):
        dict_dim = 10
        word_dim = 8
        hidden_dim = 8

        def test_old_rnn():
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
                data1 = conf_helps.data_layer(name="word", size=dict_dim)
                embd = conf_helps.embedding_layer(input=data1, size=word_dim)
                conf_helps.recurrent_group(name="rnn", step=step, input=embd)

            return str(parse_network(test))

        def test_new_rnn():
            def new_step(y):
                mem = layer.memory(name="rnn_state", size=hidden_dim)
                out = layer.fc(input=[mem],
                               step_input=y,
                               size=hidden_dim,
                               act=activation.Tanh(),
                               bias_attr=True,
                               name="rnn_state")
                return out.to_proto(dict())

            data1 = layer.data(
                name="word", type=data_type.integer_value(dict_dim))
            embd = layer.embedding(input=data1, size=word_dim)
            rnn_layer = layer.recurrent_group(
                name="rnn", step=new_step, input=embd)
            return str(layer.parse_network(rnn_layer))

        diff = difflib.unified_diff(test_old_rnn().splitlines(1),
                                    test_new_rnn().splitlines(1))
        print ''.join(diff)


if __name__ == '__main__':
    unittest.main()
