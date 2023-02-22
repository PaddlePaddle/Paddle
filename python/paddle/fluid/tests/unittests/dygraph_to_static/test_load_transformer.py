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

from __future__ import print_function

import unittest

import numpy as np

import paddle


class FakeNet:
    def __init__(self):
        self.var = paddle.to_tensor([2.0])


f = FakeNet()
g = paddle.to_tensor([1.0])


class Net(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # unsupport g as store.
        t = g * 2 + x
        t = f.var * t
        return t


class TestFallback(unittest.TestCase):
    def setUp(self):
        self.x = paddle.to_tensor(1.0).astype('int')

    def test_name_load(self):
        net_dy = Net()
        net_st = Net()
        output_dy = net_dy(self.x)
        output_st = paddle.jit.to_static(net_st)(self.x)
        np.testing.assert_allclose(output_dy.numpy(), output_st.numpy())


class TestLoad2(unittest.TestCase):
    def test_name_load_nograd(self):
        @paddle.no_grad()
        def func(x):
            x = paddle.shape(x)
            return x

        x = paddle.to_tensor([[3, 3], [1, 1]])
        output_st = paddle.jit.to_static(func)(x)
        output_dy = func(x)
        np.testing.assert_allclose(output_dy.numpy(), output_st.numpy())


if __name__ == "__main__":
    unittest.main()
