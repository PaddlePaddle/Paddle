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

import unittest

import numpy as np
from op_test import OpTest

import paddle

np.random.seed(100)
paddle.seed(100)


class TestSumAsOp(OpTest):
    def setUp(self):
        self.init_dtype()
        self.init_shape()
        self.init_input()
        self.init_attrs()
        self.calc_output()

        self.python_api = paddle.sum_as
        self.public_python_api = paddle.sum_as
        self.op_type = "sum_as"
        self.prim_op_type = "prim"
        self.inputs = {'x': self.x, 'y': self.y}
        self.outputs = {'out': self.out}
        self.if_enable_cinn()

    def init_dtype(self):
        self.dtype = np.float64

    def init_shape(self):
        self.shape_x = [30, 10, 6]
        self.shape_y = [10, 6]

    def init_input(self):
        self.x = np.random.random(self.shape_x).astype(self.dtype)
        self.y = np.random.random(self.shape_y).astype(self.dtype)

    def init_attrs(self):
        self.attrs = {'dim': [0]}

    def if_enable_cinn(self):
        pass

    def calc_output(self):
        self.out = self.x.sum(axis=tuple(self.attrs['dim']))

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(
            ['x'],
            'out',
        )


class TestSumAsOp2(TestSumAsOp):
    def init_type(self):
        self.dtype = 'float32'


class TestSumAsOp3(TestSumAsOp):
    def init_type(self):
        self.dtype = 'float16'


class TestSumAsOp4(TestSumAsOp):
    def init_type(self):
        self.dtype = 'uint16'


class TestSumAsOp5(TestSumAsOp):
    def init_type(self):
        self.dtype = 'int16'


class TestSumAsOp6(TestSumAsOp):
    def init_type(self):
        self.dtype = 'int64'


class TestSumAsOp7(TestSumAsOp):
    def init_type(self):
        self.dtype = 'bool'


class TestSumAsOp8(TestSumAsOp):
    def init_type(self):
        self.dtype = 'int32'


class TestSumAsOp9(TestSumAsOp):
    def init_shape(self):
        self.shape_x = [30, 10, 6]
        self.shape_y = [6]

    def init_attrs(self):
        self.attrs = {'dim': [0, 1]}


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
