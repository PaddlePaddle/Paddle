#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid.core as core
from op_test import OpTest


class TestMulOp(OpTest):
    def setUp(self):
        self.op_type = "mul"
        self.use_mkldnn = False
        self.inputs = {
            'X': np.random.random((32, 84)).astype("float32"),
            'Y': np.random.random((84, 100)).astype("float32")
        }
        self.attrs = {'use_mkldnn': self.use_mkldnn}
        self.outputs = {'Out': np.dot(self.inputs['X'], self.inputs['Y'])}

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['X', 'Y'], 'Out', max_relative_error=0.5)

    def test_check_grad_ingore_x(self):
        self.check_grad(
            ['Y'], 'Out', max_relative_error=0.5, no_grad_set=set("X"))

    def test_check_grad_ingore_y(self):
        self.check_grad(
            ['X'], 'Out', max_relative_error=0.5, no_grad_set=set('Y'))


class TestMulOp2(OpTest):
    def setUp(self):
        self.op_type = "mul"
        self.use_mkldnn = False
        self.inputs = {
            'X': np.random.random((15, 4, 12, 10)).astype("float32"),
            'Y': np.random.random((4, 30, 8, 2, 9)).astype("float32")
        }
        self.attrs = {
            'x_num_col_dims': 2,
            'y_num_col_dims': 2,
            'use_mkldnn': self.use_mkldnn
        }
        result = np.dot(self.inputs['X'].reshape(15 * 4, 12 * 10),
                        self.inputs['Y'].reshape(4 * 30, 8 * 2 * 9))
        result = result.reshape(15, 4, 8, 2, 9)
        self.outputs = {'Out': result}

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['X', 'Y'], 'Out', max_relative_error=0.5)

    def test_check_grad_ingore_x(self):
        self.check_grad(
            ['Y'], 'Out', max_relative_error=0.5, no_grad_set=set('X'))

    def test_check_grad_ignore_y(self):
        self.check_grad(
            ['X'], 'Out', max_relative_error=0.5, no_grad_set=set('Y'))


class TestFP16MulOp1(OpTest):
    def setUp(self):
        self.op_type = "mul"
        self.use_mkldnn = False
        x = np.random.random((32, 84)).astype("float16")
        y = np.random.random((84, 100)).astype("float16")
        self.inputs = {'X': x.view(np.uint16), 'Y': y.view(np.uint16)}
        self.attrs = {'use_mkldnn': self.use_mkldnn}
        self.outputs = {'Out': np.dot(x, y)}

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place):
                self.check_output_with_place(place, atol=1e-1)


class TestFP16MulOp2(OpTest):
    def setUp(self):
        self.op_type = "mul"
        self.use_mkldnn = False
        x = np.random.random((15, 4, 12, 10)).astype("float16")
        y = np.random.random((4, 30, 8, 2, 9)).astype("float16")
        self.inputs = {'X': x.view(np.uint16), 'Y': y.view(np.uint16)}
        self.attrs = {
            'x_num_col_dims': 2,
            'y_num_col_dims': 2,
            'use_mkldnn': self.use_mkldnn
        }
        result = np.dot(
            x.reshape(15 * 4, 12 * 10), y.reshape(4 * 30, 8 * 2 * 9))
        result = result.reshape(15, 4, 8, 2, 9)
        self.outputs = {'Out': result}

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place):
                self.check_output_with_place(place, atol=2e-1)


if __name__ == "__main__":
    unittest.main()
