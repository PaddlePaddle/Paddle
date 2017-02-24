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
import paddle.v2.data_type as data_type
import paddle.v2.layer as layer
from paddle.trainer_config_helpers.config_parser_utils import \
    parse_network_config as parse_network


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
            aaa = layer.recurrent_group(name="rnn", step=new_step, input=embd)
            return str(layer.parse_network(aaa))

        diff = difflib.unified_diff(test_old_rnn().splitlines(1),
                                    test_new_rnn().splitlines(1))
        print ''.join(diff)


if __name__ == '__main__':
    unittest.main()
