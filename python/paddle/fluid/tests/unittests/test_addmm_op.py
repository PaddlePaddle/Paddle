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

from __future__ import print_function

import unittest
import numpy as np
import paddle
import paddle.fluid.core as core
from op_test import OpTest
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard


class TestAddMMOp(OpTest):
    # test basic
    def setUp(self):
        self.op_type = "addmm"
        self.dtype = np.float64
        self.init_dtype_type()
        self.inputs = {
            'Input': np.random.random((100, 1)).astype(self.dtype),
            'X': np.random.random((100, 10)).astype(self.dtype),
            'Y': np.random.random((10, 20)).astype(self.dtype),
        }
        self.outputs = {
            'Out':
            self.inputs['Input'] + np.dot(self.inputs['X'], self.inputs['Y'])
        }

    def init_dtype_type(self):
        pass

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['Input', 'X', 'Y'], 'Out')

    def test_check_grad_x(self):
        self.check_grad(['X'], 'Out', no_grad_set=None)

    def test_check_grad_y(self):
        self.check_grad(['Y'], 'Out', no_grad_set=None)

    def test_check_grad_input(self):
        self.check_grad(['Input'], 'Out', no_grad_set=None)


class TestAddMMOpError(unittest.TestCase):
    # test error
    def test_errors(self):
        with program_guard(Program(), Program()):
            # The input type of addmm_op must be Variable.
            input = fluid.create_lod_tensor(
                np.array([[-1]]), [[1]], fluid.CPUPlace())
            x1 = fluid.create_lod_tensor(
                np.array([[-1]]), [[1]], fluid.CPUPlace())
            x2 = fluid.create_lod_tensor(
                np.array([[-1]]), [[1]], fluid.CPUPlace())
            self.assertRaises(TypeError, paddle.addmm, input, x1, x2)
            # The input dtype of mul_op must be float32 or float64.
            input = fluid.layers.data(name='input', shape=[4], dtype="int32")
            x3 = fluid.layers.data(name='x3', shape=[4], dtype="int32")
            x4 = fluid.layers.data(name='x4', shape=[4], dtype="int32")
            self.assertRaises(TypeError, paddle.addmm, input, x3, x4)


class TestAddMMOp2(TestAddMMOp):
    # test alpha and beta
    def setUp(self):
        self.op_type = "addmm"
        self.dtype = np.float64
        self.init_dtype_type()
        self.inputs = {
            'Input': np.random.random((20, 30)).astype(self.dtype),
            'X': np.random.random((20, 6)).astype(self.dtype),
            'Y': np.random.random((6, 30)).astype(self.dtype),
        }
        self.attrs = {
            'Alpha': 0.1,
            'Beta': 1.0,
        }
        self.outputs = {'Out': self.attrs['Beta'] * self.inputs['Input'] + \
                        self.attrs['Alpha'] * np.dot(self.inputs['X'], self.inputs['Y'])}


class TestAddMMOp3(OpTest):
    # test broadcast
    def setUp(self):
        self.op_type = "addmm"
        self.dtype = np.float64
        self.init_dtype_type()
        self.inputs = {
            'Input': np.random.random((1, 100)).astype(self.dtype),
            'X': np.random.random((20, 10)).astype(self.dtype),
            'Y': np.random.random((10, 100)).astype(self.dtype),
        }
        self.attrs = {
            'Alpha': 0.5,
            'Beta': 2.0,
        }
        self.outputs = {'Out': self.attrs['Beta'] * self.inputs['Input'] + \
                        self.attrs['Alpha'] * np.dot(self.inputs['X'], self.inputs['Y'])}

    def init_dtype_type(self):
        pass

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['Input', 'X', 'Y'], 'Out')

    def test_check_grad_x(self):
        self.check_grad(['X'], 'Out', no_grad_set=None)

    def test_check_grad_y(self):
        self.check_grad(['Y'], 'Out', no_grad_set=None)

    def test_check_grad_input(self):
        self.check_grad(['Input'], 'Out', no_grad_set=None)


class TestAddMMOp4(unittest.TestCase):
    def test_api_with_dygraph(self):
        np_input = np.random.random((20, 30)).astype(np.float32)
        np_x = np.random.random((20, 6)).astype(np.float32)
        np_y = np.random.random((6, 30)).astype(np.float32)

        with fluid.dygraph.guard():
            input = fluid.dygraph.to_variable(np_input)
            x = fluid.dygraph.to_variable(np_x)
            y = fluid.dygraph.to_variable(np_y)
            out = paddle.tensor.addmm(input, x, y)
            assert np.allclose(np_input + np.dot(np_x, np_y), out.numpy())


if __name__ == "__main__":
    unittest.main()
