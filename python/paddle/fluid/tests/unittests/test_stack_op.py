# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import unittest
import paddle.fluid as fluid
from op_test import OpTest


class TestStackOpBase(OpTest):
    def initDefaultParameters(self):
        self.num_inputs = 4
        self.input_dim = (5, 6, 7)
        self.axis = 0
        self.dtype = 'float64'

    def initParameters(self):
        pass

    def get_x_names(self):
        x_names = []
        for i in range(self.num_inputs):
            x_names.append('x{}'.format(i))
        return x_names

    def setUp(self):
        self.initDefaultParameters()
        self.initParameters()
        self.op_type = 'stack'
        self.x = []
        for i in range(self.num_inputs):
            self.x.append(
                np.random.random(size=self.input_dim).astype(self.dtype))

        tmp = []
        x_names = self.get_x_names()
        for i in range(self.num_inputs):
            tmp.append((x_names[i], self.x[i]))

        self.inputs = {'X': tmp}
        self.outputs = {'Y': np.stack(self.x, axis=self.axis)}
        self.attrs = {'axis': self.axis}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(self.get_x_names(), 'Y')


class TestStackOp1(TestStackOpBase):
    def initParameters(self):
        self.num_inputs = 16


class TestStackOp2(TestStackOpBase):
    def initParameters(self):
        self.num_inputs = 20


class TestStackOp3(TestStackOpBase):
    def initParameters(self):
        self.axis = -1


class TestStackOp4(TestStackOpBase):
    def initParameters(self):
        self.axis = -4


class TestStackOp5(TestStackOpBase):
    def initParameters(self):
        self.axis = 1


class TestStackOp6(TestStackOpBase):
    def initParameters(self):
        self.axis = 3


class TestStackAPIWithLoDTensorArray(unittest.TestCase):
    """
    Test stack api when the input(x) is a LoDTensorArray.
    """

    def setUp(self):
        self.axis = 1
        self.iter_num = 3
        self.input_shape = [2, 3]
        self.x = np.random.random(self.input_shape).astype("float32")
        self.place = fluid.CUDAPlace(0) \
            if fluid.is_compiled_with_cuda() else fluid.CPUPlace()
        self.set_program()

    def set_program(self):
        self.program = fluid.Program()
        with fluid.program_guard(self.program):
            input = fluid.layers.assign(self.x)
            tensor_array = fluid.layers.create_array(dtype='float32')
            zero = fluid.layers.fill_constant(shape=[1], value=0, dtype="int64")

            for i in range(self.iter_num):
                fluid.layers.array_write(input, zero + i, tensor_array)

            self.out_var = fluid.layers.stack(tensor_array, axis=self.axis)

    def test_case(self):
        self.assertTrue(self.out_var.shape[self.axis] == -1)
        exe = fluid.Executor(self.place)
        res = exe.run(self.program, fetch_list=self.out_var)
        self.assertTrue(
            np.array_equal(
                res[0], np.stack(
                    [self.x] * self.iter_num, axis=self.axis)))


if __name__ == '__main__':
    unittest.main()
