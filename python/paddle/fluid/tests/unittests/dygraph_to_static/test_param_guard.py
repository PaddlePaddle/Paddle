# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import numpy as np
import unittest

from paddle.jit import to_static, ProgramTranslator


class NetWithParameterList(paddle.nn.Layer):

    def __init__(self, in_size, out_size):
        super(NetWithParameterList, self).__init__()
        weight = self.create_parameter([in_size, out_size])
        bias = self.create_parameter([out_size], is_bias=True)
        self.params = paddle.nn.ParameterList([weight, bias])

    @to_static
    def forward(self, x):
        out = paddle.matmul(x, self.params[0])
        out = paddle.add(out, self.params[1])
        out = paddle.tanh(out)
        return out


class NetWithParameterListIter(NetWithParameterList):

    def __init__(self, in_size, out_size):
        super(NetWithParameterListIter, self).__init__(in_size, out_size)

    @to_static
    def forward(self, x):
        # NOTE: manually trigger `__iter__` logic.
        params = list(self.params.__iter__())
        out = paddle.matmul(x, params[0])
        out = paddle.add(out, params[1])
        out = paddle.tanh(out)
        return out


class TestParameterList(unittest.TestCase):

    def setUp(self):
        self.seed = 2021
        self.iter_num = 5
        self.prog_trans = ProgramTranslator()

    def train(self, is_iter, to_static):
        paddle.seed(self.seed)
        np.random.seed(self.seed)
        self.prog_trans.enable(to_static)
        if is_iter:
            net = NetWithParameterList(10, 3)
        else:
            net = NetWithParameterListIter(10, 3)
        sgd = paddle.optimizer.SGD(0.1, parameters=net.parameters())

        for batch_id in range(self.iter_num):
            x = paddle.rand([4, 10], dtype='float32')
            out = net(x)
            loss = paddle.mean(out)
            loss.backward()
            sgd.step()
            sgd.clear_grad()

        return loss

    def test_parameter_list(self):
        static_loss = self.train(False, to_static=True)
        dygraph_loss = self.train(False, to_static=False)
        np.testing.assert_allclose(dygraph_loss, static_loss, rtol=1e-05)

    def test_parameter_list_iter(self):
        static_loss = self.train(True, to_static=True)
        dygraph_loss = self.train(True, to_static=False)
        np.testing.assert_allclose(dygraph_loss, static_loss, rtol=1e-05)


class NetWithRawParamList(paddle.nn.Layer):

    def __init__(self, in_size, out_size):
        super(NetWithRawParamList, self).__init__()
        weight = self.add_parameter('w',
                                    self.create_parameter([in_size, out_size]))
        bias = self.add_parameter(
            'b', self.create_parameter([out_size], is_bias=True))
        self.params = [weight]
        self.bias_dict = {'b': bias}

    @to_static
    def forward(self, x):
        out = paddle.matmul(x, self.params[0])
        out = paddle.add(out, self.bias_dict['b'])
        out = paddle.tanh(out)
        return out


class TestRawParameterList(unittest.TestCase):

    def setUp(self):
        self.seed = 2021
        self.iter_num = 5
        self.prog_trans = ProgramTranslator()

    def init_net(self):
        self.net = NetWithRawParamList(10, 3)

    def train(self, to_static):
        paddle.seed(self.seed)
        np.random.seed(self.seed)
        self.prog_trans.enable(to_static)
        self.init_net()

        sgd = paddle.optimizer.SGD(0.1, parameters=self.net.parameters())

        for batch_id in range(self.iter_num):
            x = paddle.rand([4, 10], dtype='float32')
            out = self.net(x)
            loss = paddle.mean(out)
            loss.backward()
            sgd.step()
            sgd.clear_grad()

        return loss

    def test_parameter_list(self):
        static_loss = self.train(to_static=True)
        dygraph_loss = self.train(to_static=False)
        np.testing.assert_allclose(dygraph_loss, static_loss, rtol=1e-05)


class NetWithSubLayerParamList(paddle.nn.Layer):

    def __init__(self, sub_layer):
        super(NetWithSubLayerParamList, self).__init__()
        self.sub_layer = sub_layer
        self.params = [sub_layer.weight]
        self.bias_dict = {'b': sub_layer.bias}

    @to_static
    def forward(self, x):
        out = paddle.matmul(x, self.params[0])
        out = paddle.add(out, self.bias_dict['b'])
        out = paddle.tanh(out)
        return out


class TestSubLayerParameterList(TestRawParameterList):

    def init_net(self):
        fc = paddle.nn.Linear(10, 3)
        self.net = NetWithSubLayerParamList(fc)


if __name__ == '__main__':
    unittest.main()
