#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
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
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.op import Operator
from paddle.fluid import compiler, Program, program_guard
from op_test import OpTest, skip_check_grad_ci


class AddcmulOp(OpTest):
    def setUp(self):
        self.op_type = "addcmul"
        self.init_dtype()
        self.init_value()
        self.init_shape()
        self.init_input_output()

        self.inputs = {
            'Input': OpTest.np_dtype_to_fluid_dtype(self.input),
            'Tensor1': OpTest.np_dtype_to_fluid_dtype(self.tensor1),
            'Tensor2': OpTest.np_dtype_to_fluid_dtype(self.tensor2)
        }
        self.outputs = {'Out': self.out}
        self.attrs = {'value': self.value}

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['Input', 'Tensor1', 'Tensor2'], 'Out')

    def init_input_output(self):
        self.input = np.random.uniform(0.1, 1,
                                       self.input_shape).astype(self.dtype)
        self.tensor1 = np.random.uniform(0.1, 1,
                                         self.tensor1_shape).astype(self.dtype)
        self.tensor2 = np.random.uniform(0.1, 1,
                                         self.tensor2_shape).astype(self.dtype)
        self.out = np.add(self.input,
                          np.multiply(self.tensor1, self.tensor2) * self.value)

    def init_value(self):
        self.value = 1.0

    def init_dtype(self):
        self.dtype = np.float64

    def init_shape(self):
        self.input_shape = [16, 12]
        self.tensor1_shape = [16, 12]
        self.tensor2_shape = [16, 12]


class TestAddcmulOp_value(AddcmulOp):
    def init_value(self):
        self.value = 0.3


class TestAddcmulOp_Vector(AddcmulOp):
    def init_shape(self):
        self.input_shape = [100]
        self.tensor1_shape = [100]
        self.tensor2_shape = [100]


class TestAddcmulOp_with_broadcast_0(AddcmulOp):
    def init_shape(self):
        self.input_shape = [2, 100]
        self.tensor1_shape = [2, 100]
        self.tensor2_shape = [100]


class TestAddcmulOp_with_broadcast_1(AddcmulOp):
    def init_shape(self):
        self.input_shape = [4, 100]
        self.tensor1_shape = [100]
        self.tensor2_shape = [4, 100]


class TestAddcmulOp_with_broadcast_2(AddcmulOp):
    def init_shape(self):
        self.input_shape = [3, 100]
        self.tensor1_shape = [3, 100]
        self.tensor2_shape = [100]


class TestAddcmulOp_with_broadcast_3(AddcmulOp):
    def init_shape(self):
        self.input_shape = [100, 4]
        self.tensor1_shape = [100, 4]
        self.tensor2_shape = [100, 1]


class TestAddcmulOp_with_broadcast_4(AddcmulOp):
    def init_shape(self):
        self.input_shape = [100, 2]
        self.tensor1_shape = [100, 1]
        self.tensor2_shape = [100, 1]


class TestAddcmulOp_multidim_0(AddcmulOp):
    def init_shape(self):
        self.input_shape = [50, 2, 4, 2]
        self.tensor1_shape = [50, 2, 4, 2]
        self.tensor2_shape = [50, 1, 1, 2]


class TestAddcmulOp_multidim_1(AddcmulOp):
    def init_shape(self):
        self.input_shape = [1, 2, 2, 50]
        self.tensor1_shape = [1, 2, 2, 50]
        self.tensor2_shape = [1, 2, 1, 50]


class TestAddcmulOp_different_size(AddcmulOp):
    def init_shape(self):
        self.input_shape = [2, 2, 10, 10]
        self.tensor1_shape = [10, 10]
        self.tensor2_shape = [2, 2, 10, 10]

    def init_input_output(self):
        self.input = np.random.uniform(0.1, 1,
                                       self.input_shape).astype(self.dtype)
        self.tensor1 = np.random.uniform(0.1, 1,
                                         self.tensor1_shape).astype(self.dtype)
        self.tensor2 = np.random.uniform(0.1, 1,
                                         self.tensor2_shape).astype(self.dtype)
        self.out = np.add(self.input,
                          np.multiply(
                              self.tensor1.reshape(1, 1, 10, 10),
                              self.tensor2) * self.value)


class TestAddcmulLayer(unittest.TestCase):
    def test_addcmul_layer(self):
        program = Program()
        with program_guard(program):
            data_shape = [3, 64, 64]
            input = fluid.data(name='in', shape=data_shape, dtype='float32')
            tensor1 = fluid.data(name='t1', shape=data_shape, dtype='float32')
            tensor2 = fluid.data(name='t2', shape=data_shape, dtype='float32')

            out = paddle.addcmul(input, tensor1, tensor2)
            self.assertEqual(out.shape, input.shape)


class InvalidInputTest(unittest.TestCase):
    def test_error(self):
        def test_invalid_input():
            input = [20, 20]
            tensor1 = fluid.data(
                name='tensor1', shape=[20, 20], dtype='float32')
            tensor2 = fluid.data(
                name='tensor2', shape=[20, 20], dtype='float32')
            out = paddle.addcmul(input, tensor1, tensor2)

        self.assertRaises(TypeError, test_invalid_input)

        def test_invalid_tensor1():
            input = fluid.data(name='input', shape=[20, 20], dtype='float32')
            tensor1 = [20, 20]
            tensor2 = fluid.data(
                name='tensor3', shape=[20, 20], dtype='float32')
            out = paddle.addcmul(input, tensor1, tensor2)

        self.assertRaises(TypeError, test_invalid_tensor1)

        def test_invalid_tensor2():
            input = fluid.data(name='input1', shape=[20, 20], dtype='float32')
            tensor1 = fluid.data(
                name='tensor4', shape=[20, 20], dtype='float32')
            tensor2 = [20, 20]
            out = paddle.addcmul(input, tensor1, tensor2)

        self.assertRaises(TypeError, test_invalid_tensor2)


if __name__ == '__main__':
    unittest.main()
