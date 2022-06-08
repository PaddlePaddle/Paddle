#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.fluid.core as core
import unittest
import numpy as np
from op_test import OpTest


class TestFillAnyOp(OpTest):

    def setUp(self):
        self.op_type = "fill_any"
        self.dtype = 'float64'
        self.value = 0.0
        self.init()
        self.inputs = {'X': np.random.random((20, 30)).astype(self.dtype)}
        self.attrs = {
            'value_float': float(self.value),
            'value_int': int(self.value)
        }
        self.outputs = {
            'Out':
            self.value * np.ones_like(self.inputs["X"]).astype(self.dtype)
        }

    def init(self):
        pass

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X"], "Out")


class TestFillAnyOpFloat32(TestFillAnyOp):

    def init(self):
        self.dtype = np.float32
        self.value = 0.0


class TestFillAnyOpFloat16(TestFillAnyOp):

    def init(self):
        self.dtype = np.float16


class TestFillAnyOpvalue1(TestFillAnyOp):

    def init(self):
        self.dtype = np.float32
        self.value = 111111555


class TestFillAnyOpvalue2(TestFillAnyOp):

    def init(self):
        self.dtype = np.float32
        self.value = 11111.1111


if __name__ == "__main__":
    unittest.main()
