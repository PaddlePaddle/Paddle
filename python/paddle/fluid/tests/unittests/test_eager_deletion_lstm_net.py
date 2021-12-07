# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from test_eager_deletion_dynamic_rnn_base import TestBase
import paddle.fluid as fluid
import unittest

fluid.core._set_eager_deletion_mode(0.0, 1.0, True)


def lstm_net(data,
             label,
             dict_dim,
             emb_dim=128,
             hid_dim=128,
             hid_dim2=96,
             class_dim=2,
             emb_lr=30.0):
    emb = fluid.layers.embedding(
        input=data,
        size=[dict_dim, emb_dim],
        param_attr=fluid.ParamAttr(learning_rate=emb_lr))
    fc0 = fluid.layers.fc(input=emb, size=hid_dim * 4)
    lstm_h, c = fluid.layers.dynamic_lstm(
        input=fc0, size=hid_dim * 4, is_reverse=False)
    lstm_max = fluid.layers.sequence_pool(input=lstm_h, pool_type='max')
    lstm_max_tanh = fluid.layers.tanh(lstm_max)
    fc1 = fluid.layers.fc(input=lstm_max_tanh, size=hid_dim2, act='tanh')
    prediction = fluid.layers.fc(input=fc1, size=class_dim, act='softmax')
    cost = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_cost = fluid.layers.mean(x=cost)
    return avg_cost


class LSTMTest(TestBase):
    def setUp(self):
        self.net = lstm_net


if __name__ == "__main__":
    unittest.main()
