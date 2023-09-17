# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import unittest

import paddle
from paddle.distributed.fleet.utils import recompute


class Layer(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.linear1 = paddle.nn.Linear(10, 10)
        self.linear2 = paddle.nn.Linear(10, 10)
        self.linear3 = paddle.nn.Linear(10, 10)
        self.silu1 = paddle.nn.Silu()
        self.silu2 = paddle.nn.Silu()
        self.silu3 = paddle.nn.Silu()

    def forward(self, x, y):
        assert type(x) is tuple
        assert len(x) == 2
        o1 = self.silu1(self.linear1(x[0]))
        o2 = self.silu2(self.linear2(x[1]))
        o3 = self.silu3(self.linear3(y))
        o = o1 + o2 + o3
        return o


class TestPyLayer(unittest.TestCase):
    def test_tuple_input(self):
        layer = Layer()
        x1 = paddle.rand(shape=[10, 10])
        x1.stop_gradient = False
        x2 = paddle.rand(shape=[10, 10])
        x2.stop_gradient = False
        y = paddle.rand(shape=[10, 10])
        y.stop_gradient = False
        o = recompute(layer, (x1, x2), y)
        loss = paddle.mean(o, keepdim=True)
        loss.backward()

    def test_tuple_input_with_non_tensor(self):
        layer = Layer()
        x1 = paddle.rand(shape=[10, 10])
        x1.stop_gradient = False
        y = paddle.rand(shape=[10, 10])
        y.stop_gradient = False
        try:
            o = recompute(layer, (x1, True), y)
        except ValueError:
            pass

    def test_tuple_input_with_different_stop_gradient(self):
        layer = Layer()
        x1 = paddle.rand(shape=[10, 10])
        x1.stop_gradient = False
        x2 = paddle.rand(shape=[10, 10])
        y = paddle.rand(shape=[10, 10])
        y.stop_gradient = False
        try:
            o = recompute(layer, (x1, True), y)
        except ValueError:
            pass

    def test_tuple_input_all_no_gradient(self):
        layer = Layer()
        x1 = paddle.rand(shape=[10, 10])
        x2 = paddle.rand(shape=[10, 10])
        y = paddle.rand(shape=[10, 10])
        y.stop_gradient = False
        o = recompute(layer, (x1, x2), y)
        loss = paddle.mean(o, keepdim=True)
        loss.backward()


if __name__ == '__main__':
    unittest.main()
