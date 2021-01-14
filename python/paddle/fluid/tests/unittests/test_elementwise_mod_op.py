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
        self.op_type = "elementwise_mod"
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
        self.check_output()

    def init_input_output(self):
        self.x = np.random.uniform(0, 10000, [10, 10]).astype(self.dtype)
        self.y = np.random.uniform(0, 1000, [10, 10]).astype(self.dtype)
        self.out = np.mod(self.x, self.y)

    def init_dtype(self):
        self.dtype = np.int32

    def init_axis(self):
        pass


class TestElementwiseModOp_scalar(TestElementwiseModOp):
    def init_input_output(self):
        scale_x = random.randint(0, 100000000)
        scale_y = random.randint(1, 100000000)
        self.x = (np.random.rand(2, 3, 4) * scale_x).astype(self.dtype)
        self.y = (np.random.rand(1) * scale_y + 1).astype(self.dtype)
        self.out = np.mod(self.x, self.y)


class TestElementwiseModOpFloat(TestElementwiseModOp):
    def init_dtype(self):
        self.dtype = np.float32

    def init_input_output(self):
        self.x = np.random.uniform(-1000, 1000, [10, 10]).astype(self.dtype)
        self.y = np.random.uniform(-100, 100, [10, 10]).astype(self.dtype)
        self.out = np.fmod(self.y + np.fmod(self.x, self.y), self.y)

    def test_check_output(self):
        self.check_output()


class TestElementwiseModOpDouble(TestElementwiseModOpFloat):
    def init_dtype(self):
        self.dtype = np.float64


class TestRemainderOp(unittest.TestCase):
    def test_name(self):
        with fluid.program_guard(fluid.Program()):
            x = fluid.data(name="x", shape=[2, 3], dtype="int64")
            y = fluid.data(name='y', shape=[2, 3], dtype='int64')

            y_1 = paddle.remainder(x, y, name='div_res')
            self.assertEqual(('div_res' in y_1.name), True)

    def test_dygraph(self):
        with fluid.dygraph.guard():
            np_x = np.array([2, 3, 8, 7]).astype('int64')
            np_y = np.array([1, 5, 3, 3]).astype('int64')
            x = paddle.to_tensor(np_x)
            y = paddle.to_tensor(np_y)
            z = paddle.remainder(x, y)
            np_z = z.numpy()
            z_expected = np.array([0, 3, 2, 1])
            self.assertEqual((np_z == z_expected).all(), True)

            np_x = np.array([-3.3, 11.5, -2, 3.5])
            np_y = np.array([-1.2, 2., 3.3, -2.3])
            x = paddle.to_tensor(np_x)
            y = paddle.to_tensor(np_y)
            z = x % y
            z_expected = np.array([-0.9, 1.5, 1.3, -1.1])
            self.assertEqual(np.allclose(z_expected, z.numpy()), True)

            np_x = np.array([-3, 11, -2, 3])
            np_y = np.array([-1, 2, 3, -2])
            x = paddle.to_tensor(np_x, dtype="int64")
            y = paddle.to_tensor(np_y, dtype="int64")
            z = x % y
            z_expected = np.array([0, 1, 1, -1])
            self.assertEqual(np.allclose(z_expected, z.numpy()), True)


if __name__ == '__main__':
    unittest.main()
