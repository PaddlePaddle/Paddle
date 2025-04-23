# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import paddle


class TestInitParamsDecorator(unittest.TestCase):
    def setUp(self):
        paddle.seed(2023)
        np.random.seed(2023)

    def test_init_params_decorator(self):
        # Implicit initialization
        class LazyInitLayer(paddle.nn.Layer):
            def __init__(self):
                super().__init__()
                self.weight = self.create_parameter(shape=[10, 5])

            def forward(self, x):
                return paddle.matmul(x, self.weight)

        with paddle.LazyGuard():
            layer = LazyInitLayer()

        self.assertFalse(layer.weight._is_initialized())

        x = paddle.randn([4, 10])
        output = layer(x)

        self.assertTrue(layer.weight._is_initialized())

        # Explicit initialization
        class ManualInitLayer(paddle.nn.Layer):
            def __init__(self):
                super().__init__()
                self.weight = self.create_parameter(shape=[10, 5])
                self.bias = self.create_parameter(shape=[5], is_bias=True)

            def forward(self, x):
                return paddle.matmul(x, self.weight) + self.bias

        with paddle.LazyGuard():
            layer = ManualInitLayer()

        self.assertFalse(layer.weight._is_initialized())
        self.assertFalse(layer.bias._is_initialized())

        for p in layer.parameters():
            if not p._is_initialized():
                p.initialize()

        self.assertTrue(layer.weight._is_initialized())
        self.assertTrue(layer.bias._is_initialized())


if __name__ == '__main__':
    unittest.main()
