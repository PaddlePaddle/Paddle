#  Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import unittest
import sys
sys.path.append("..")
from op_test import OpTest, skip_check_grad_ci
import paddle
import paddle.fluid as fluid
from paddle.fluid.core import ops

paddle.enable_static()
SEED = 2022


class TestElementwiseDiv(OpTest):
    def setUp(self):
        self.set_mlu()
        self.op_type = "elementwise_div"

        self.init_dtype()
        np.random.seed(SEED)
        x = np.random.uniform(1, 2, [11, 17]).astype(self.dtype)
        y = np.random.uniform(1, 2, [11, 17]).astype(self.dtype)
        out = np.divide(x, y)

        self.inputs = {
            'X': OpTest.np_dtype_to_fluid_dtype(x),
            'Y': OpTest.np_dtype_to_fluid_dtype(y)
        }
        self.attrs = {}
        self.outputs = {'Out': out}

    def set_mlu(self):
        self.__class__.use_mlu = True
        self.place = paddle.device.MLUPlace(0)

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad_normal(self):
        self.check_grad_with_place(
            self.place, ['X', 'Y'], 'Out', max_relative_error=0.05)

    def test_check_grad_ingore_x(self):
        self.check_grad_with_place(
            self.place, ['Y'],
            'Out',
            max_relative_error=0.05,
            no_grad_set=set("X"))

    def test_check_grad_ingore_y(self):
        self.check_grad_with_place(
            self.place, ['X'],
            'Out',
            max_relative_error=0.05,
            no_grad_set=set("Y"))


class TestElementwiseDivFp16(OpTest):
    def setUp(self):
        self.set_mlu()
        self.op_type = "elementwise_div"

        self.init_dtype()
        np.random.seed(SEED)
        x = np.random.uniform(1, 2, [3, 4]).astype(self.dtype)
        y = np.random.uniform(1, 2, [3, 4]).astype(self.dtype)
        out = np.divide(x, y)

        self.inputs = {
            'X': OpTest.np_dtype_to_fluid_dtype(x),
            'Y': OpTest.np_dtype_to_fluid_dtype(y)
        }
        self.attrs = {}
        self.outputs = {'Out': out}

    def set_mlu(self):
        self.__class__.use_mlu = True
        self.__class__.no_need_check_grad = True
        self.place = paddle.device.MLUPlace(0)

    def init_dtype(self):
        self.dtype = np.float16

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-5)


@skip_check_grad_ci(
    reason="[skip shape check] Use y_shape(1) to test broadcast.")
class TestTestElementwiseDiv_scalar(TestElementwiseDiv):
    def setUp(self):
        self.set_mlu()
        self.op_type = "elementwise_div"
        self.inputs = {
            'X': np.random.uniform(0.1, 1, [20, 3, 4]).astype(np.float32),
            'Y': np.random.uniform(0.1, 1, [1]).astype(np.float32)
        }
        self.outputs = {'Out': self.inputs['X'] / self.inputs['Y']}


class TestTestElementwiseDiv_Vector(TestElementwiseDiv):
    def setUp(self):
        self.set_mlu()
        self.op_type = "elementwise_div"
        self.inputs = {
            'X': np.random.uniform(0.1, 1, [100]).astype("float32"),
            'Y': np.random.uniform(0.1, 1, [100]).astype("float32")
        }
        self.outputs = {'Out': np.divide(self.inputs['X'], self.inputs['Y'])}


class TestTestElementwiseDiv_broadcast_0(TestElementwiseDiv):
    def setUp(self):
        self.set_mlu()
        self.op_type = "elementwise_div"
        self.inputs = {
            'X': np.random.uniform(0.1, 1, [100, 3, 4]).astype("float32"),
            'Y': np.random.uniform(0.1, 1, [100]).astype("float32")
        }

        self.attrs = {'axis': 0}
        self.outputs = {
            'Out':
            np.divide(self.inputs['X'], self.inputs['Y'].reshape(100, 1, 1))
        }


class TestTestElementwiseDiv_broadcast_1(TestElementwiseDiv):
    def setUp(self):
        self.set_mlu()
        self.op_type = "elementwise_div"
        self.inputs = {
            'X': np.random.uniform(0.1, 1, [2, 100, 4]).astype("float32"),
            'Y': np.random.uniform(0.1, 1, [100]).astype("float32")
        }

        self.attrs = {'axis': 1}
        self.outputs = {
            'Out':
            np.divide(self.inputs['X'], self.inputs['Y'].reshape(1, 100, 1))
        }


class TestTestElementwiseDiv_broadcast_2(TestElementwiseDiv):
    def setUp(self):
        self.set_mlu()
        self.op_type = "elementwise_div"
        self.inputs = {
            'X': np.random.uniform(0.1, 1, [2, 3, 100]).astype("float32"),
            'Y': np.random.uniform(0.1, 1, [100]).astype("float32")
        }

        self.outputs = {
            'Out':
            np.divide(self.inputs['X'], self.inputs['Y'].reshape(1, 1, 100))
        }


class TestTestElementwiseDiv_broadcast_3(TestElementwiseDiv):
    def setUp(self):
        self.set_mlu()
        self.op_type = "elementwise_div"
        self.inputs = {
            'X': np.random.uniform(0.1, 1, [2, 10, 12, 5]).astype("float32"),
            'Y': np.random.uniform(0.1, 1, [10, 12]).astype("float32")
        }

        self.attrs = {'axis': 1}
        self.outputs = {
            'Out':
            np.divide(self.inputs['X'], self.inputs['Y'].reshape(1, 10, 12, 1))
        }


class TestTestElementwiseDiv_broadcast_4(TestElementwiseDiv):
    def setUp(self):
        self.set_mlu()
        self.op_type = "elementwise_div"
        self.inputs = {
            'X': np.random.uniform(0.1, 1, [2, 3, 50]).astype("float32"),
            'Y': np.random.uniform(0.1, 1, [2, 1, 50]).astype("float32")
        }
        self.outputs = {'Out': np.divide(self.inputs['X'], self.inputs['Y'])}


class TestTestElementwiseDiv_broadcast_5(TestElementwiseDiv):
    def setUp(self):
        self.set_mlu()
        self.op_type = "elementwise_div"
        self.inputs = {
            'X': np.random.uniform(0.1, 1, [2, 3, 4, 20]).astype("float32"),
            'Y': np.random.uniform(0.1, 1, [2, 3, 1, 20]).astype("float32")
        }
        self.outputs = {'Out': np.divide(self.inputs['X'], self.inputs['Y'])}


class TestTestElementwiseDiv_commonuse_1(TestElementwiseDiv):
    def setUp(self):
        self.set_mlu()
        self.op_type = "elementwise_div"
        self.inputs = {
            'X': np.random.uniform(0.1, 1, [2, 3, 100]).astype("float32"),
            'Y': np.random.uniform(0.1, 1, [1, 1, 100]).astype("float32"),
        }
        self.outputs = {'Out': np.divide(self.inputs['X'], self.inputs['Y'])}


class TestTestElementwiseDiv_commonuse_2(TestElementwiseDiv):
    def setUp(self):
        self.set_mlu()
        self.op_type = "elementwise_div"
        self.inputs = {
            'X': np.random.uniform(0.1, 1, [30, 3, 1, 5]).astype("float32"),
            'Y': np.random.uniform(0.1, 1, [30, 1, 4, 1]).astype("float32"),
        }
        self.outputs = {'Out': np.divide(self.inputs['X'], self.inputs['Y'])}


class TestTestElementwiseDiv_xsize_lessthan_ysize(TestElementwiseDiv):
    def setUp(self):
        self.set_mlu()
        self.op_type = "elementwise_div"
        self.inputs = {
            'X': np.random.uniform(0.1, 1, [10, 12]).astype("float32"),
            'Y': np.random.uniform(0.1, 1, [2, 3, 10, 12]).astype("float32"),
        }

        self.attrs = {'axis': 2}

        self.outputs = {'Out': np.divide(self.inputs['X'], self.inputs['Y'])}


if __name__ == '__main__':
    unittest.main()
