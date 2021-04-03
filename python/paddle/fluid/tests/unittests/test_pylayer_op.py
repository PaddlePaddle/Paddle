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

from __future__ import print_function

import unittest
import numpy as np

import paddle
from paddle.nn.layer import PyLayer


class tanh(PyLayer):
    @staticmethod
    def backward(ctx, x1, x2, y1, y2, dy1, dy2):
        print(ctx.func)
        re1 = dy1 * (1 - paddle.square(y1))
        re2 = dy2 * (1 - paddle.square(y2))
        return re1, re2

    @staticmethod
    def forward(ctx, x1, x2, func):
        ctx.func = func
        y1 = func(x1)
        y2 = func(x2)
        return y1, y2


class TestSaveLoadLargeParameters(unittest.TestCase):
    def test_simple_pylayer(self):
        input1 = paddle.randn([2, 3]).astype("float64")
        input1.stop_gradient = False
        input2 = input1.detach().clone()
        input1.stop_gradient = False
        input2.stop_gradient = False
        # import pdb; pdb.set_trace()
        z = tanh.apply(input1, input1, paddle.tanh)
        z = z[0] + z[1]
        z.mean().backward()

        z2 = paddle.tanh(input2) + paddle.tanh(input2)
        z2.mean().backward()

        self.assertTrue(np.max(np.abs((input1.grad - input2.grad))) < 1e-10)
