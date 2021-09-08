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
import paddle.nn.functional as F
from paddle import fluid
import paddle.fluid.dygraph as dg
import paddle.fluid.initializer as I
import numpy as np
import unittest
from unittest import TestCase


class TestFunctionalConv1DError(TestCase):
    def setUp(self):
        self.input = []
        self.filter = []
        self.bias = None
        self.padding = 0
        self.stride = 1
        self.dilation = 1
        self.groups = 1
        self.data_format = "NCL"

    def dygraph_case(self):
        with dg.guard():
            x = dg.to_variable(self.input, dtype=paddle.float32)
            w = dg.to_variable(self.filter, dtype=paddle.float32)
            b = None if self.bias is None else dg.to_variable(
                self.bias, dtype=paddle.float32)
            y = F.conv1d(
                x,
                w,
                b,
                padding=self.padding,
                stride=self.stride,
                dilation=self.dilation,
                groups=self.groups,
                data_format=self.data_format)

    def test_exception(self):
        with self.assertRaises(ValueError):
            self.dygraph_case()


class TestFunctionalConv1DErrorCase1(TestFunctionalConv1DError):
    def setUp(self):
        self.input = np.random.randn(1, 3, 3)
        self.filter = np.random.randn(3, 3, 1)
        self.bias = None
        self.padding = 0
        self.stride = 1
        self.dilation = 1
        self.groups = 0
        self.data_format = "NCL"


if __name__ == "__main__":
    unittest.main()
