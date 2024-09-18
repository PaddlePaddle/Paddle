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

import numpy as np
from dygraph_to_static_utils import (
    Dy2StTestBase,
)

import paddle


class Net(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out = x + 1
        return out


class TestBackwardWithoutParams(Dy2StTestBase):
    def test_run(self):
        net = paddle.jit.to_static(Net())

        x = paddle.ones([2, 2])
        x.stop_gradient = False
        out = net(x)
        loss = paddle.mean(out)
        loss.backward()
        np.testing.assert_equal(x.grad.numpy(), np.full(x.shape, 0.25))


class ZeroSizeNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        y = paddle.randn((0,))
        out = paddle.nn.functional.relu(x)
        y.stop_gradient = True
        return y, out


class TestZeroSizeNet(Dy2StTestBase):
    def test_run(self):
        net = paddle.jit.to_static(ZeroSizeNet())
        x = paddle.ones([2, 2])
        x.stop_gradient = False
        _, out = net(x)
        loss = paddle.mean(out)
        loss.backward()
        np.testing.assert_equal(x.grad.numpy(), np.full(x.shape, 0.25))


if __name__ == '__main__':
    unittest.main()
