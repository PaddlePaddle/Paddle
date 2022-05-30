#  Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest

import random


class TestElementwiseModOp(OpTest):
    def init_kernel_type(self):
        self.use_mkldnn = False

    def setUp(self):
        self.op_type = "elementwise_floordiv"
        self.python_api = paddle.floor_divide
        self.dtype = np.int32
        self.axis = -1
        self.init_dtype()
        self.init_input_output()
        self.init_kernel_type()
        self.init_axis()

        self.inputs = {
            'X': OpTest.np_dtype_to_fluid_dtype(self.x),
            'Y': OpTest.np_dtype_to_fluid_dtype(self.y)
        }
        self.attrs = {'axis': self.axis, 'use_mkldnn': self.use_mkldnn}
        self.outputs = {'Out': self.out}

    def test_check_output(self):
        self.check_output(check_eager=True)

    def init_input_output(self):
        self.x = np.random.uniform(0, 10000, [10, 10]).astype(self.dtype)
        self.y = np.random.uniform(0, 1000, [10, 10]).astype(self.dtype)
        self.out = np.floor_divide(self.x, self.y)

    def init_dtype(self):
        pass

    def init_axis(self):
        pass


class TestElementwiseModOp_scalar(TestElementwiseModOp):
    def init_input_output(self):
        scale_x = random.randint(0, 100000000)
        scale_y = random.randint(1, 100000000)
        self.x = (np.random.rand(2, 3, 4) * scale_x).astype(self.dtype)
        self.y = (np.random.rand(1) * scale_y + 1).astype(self.dtype)
        self.out = np.floor_divide(self.x, self.y)


class TestElementwiseModOpInverse(TestElementwiseModOp):
    def init_input_output(self):
        self.x = np.random.uniform(0, 10000, [10]).astype(self.dtype)
        self.y = np.random.uniform(0, 1000, [10, 10]).astype(self.dtype)
        self.out = np.floor_divide(self.x, self.y)


class TestFloorDivideOp(unittest.TestCase):
    def test_name(self):
        with fluid.program_guard(fluid.Program()):
            x = fluid.data(name="x", shape=[2, 3], dtype="int64")
            y = fluid.data(name='y', shape=[2, 3], dtype='int64')

            y_1 = paddle.floor_divide(x, y, name='div_res')
            self.assertEqual(('div_res' in y_1.name), True)

    def test_dygraph(self):
        with fluid.dygraph.guard():
            np_x = np.array([2, 3, 8, 7]).astype('int64')
            np_y = np.array([1, 5, 3, 3]).astype('int64')
            x = paddle.to_tensor(np_x)
            y = paddle.to_tensor(np_y)
            z = paddle.floor_divide(x, y)
            np_z = z.numpy()
            z_expected = np.array([2, 0, 2, 2])
            self.assertEqual((np_z == z_expected).all(), True)

        with fluid.dygraph.guard(fluid.CPUPlace()):
            # divide by zero 
            np_x = np.array([2, 3, 4])
            np_y = np.array([0])
            x = paddle.to_tensor(np_x)
            y = paddle.to_tensor(np_y)
            try:
                z = x // y
            except Exception as e:
                print("Error: Divide by zero encounter in floor_divide\n")

            # divide by zero 
            np_x = np.array([2])
            np_y = np.array([0, 0, 0])
            x = paddle.to_tensor(np_x, dtype="int32")
            y = paddle.to_tensor(np_y, dtype="int32")
            try:
                z = x // y
            except Exception as e:
                print("Error: Divide by zero encounter in floor_divide\n")


if __name__ == '__main__':
    unittest.main()
