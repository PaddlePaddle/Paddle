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

from test_case_base import TestCaseBase, test_with_faster_guard

import paddle


class SimpleNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.linear1 = paddle.nn.Linear(10, 1)

    def forward(self, x):
        out1 = self.linear1(x)
        return out1


class SimpleNet_bound(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.linear1 = paddle.nn.Linear(10, 1)

    def add(self, x):
        return x + 1

    def forward(self, x):
        x = self.add(x)
        out1 = self.linear1(x)
        return out1


def net_call(x: paddle.Tensor, net):
    return net(x)


def net_call_passed_by_user(x: paddle.Tensor, net_forward):
    return net_forward(x)


class SimpleNetWithSequenital(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.seq = paddle.nn.Sequential(
            paddle.nn.Linear(10, 10),
            paddle.nn.Linear(10, 10),
            paddle.nn.Linear(10, 1),
        )

    def forward(self, x):
        out1 = self.seq(x)
        return out1


class TestLayer(TestCaseBase):
    @test_with_faster_guard
    def test_layer(self):
        x = paddle.rand((10,))
        y = paddle.rand((10, 10))
        net = SimpleNet()
        self.assert_results(net_call, x, net)
        self.assert_results(net_call, y, net)
        self.assert_results(net_call_passed_by_user, x, net.forward)

    @test_with_faster_guard
    def test_layer_with_sequential(self):
        x = paddle.rand((10,))
        y = paddle.rand((10, 10))
        net = SimpleNetWithSequenital()
        self.assert_results(net_call, x, net)
        self.assert_results(net_call, y, net)
        self.assert_results(net_call_passed_by_user, x, net.forward)

    @test_with_faster_guard
    def test_bound(self):
        x = paddle.rand((10,))
        y = paddle.rand((10, 10))
        net = SimpleNet_bound()
        self.assert_results(net_call, x, net)
        self.assert_results(net_call, y, net)


if __name__ == "__main__":
    unittest.main()
