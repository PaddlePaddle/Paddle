# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import annotations

import unittest

import paddle
from paddle.static.input import InputSpec


class SimpleNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.conv = paddle.nn.Conv2D(3, 3, 3)
        self.bn = paddle.nn.BatchNorm2D(3)
        self.depthwise_conv = paddle.nn.Conv2D(3, 3, 3, groups=3)

    def forward(self, x):
        z1 = self.conv(x)
        z2 = self.bn(x)
        z3 = self.depthwise_conv(x)
        return z1, z2, z3


class TestDynamicInputSpec(unittest.TestCase):
    def setUp(self):
        self.net = paddle.jit.to_static(
            SimpleNet(),
            full_graph=True,
            input_spec=[
                InputSpec(shape=[None, None, None, None], dtype='float32')
            ],
        )

    def test_dynamic_input_spec(self):
        self.net(paddle.randn([1, 3, 32, 32]))


if __name__ == '__main__':
    unittest.main()
