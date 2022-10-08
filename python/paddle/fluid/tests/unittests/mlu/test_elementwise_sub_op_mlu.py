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

import numpy as np
import unittest
import sys

sys.path.append("..")
from op_test import OpTest, skip_check_grad_ci
import paddle
import paddle.fluid as fluid

paddle.enable_static()

SEED = 2022


class TestElementwiseSubOp(OpTest):

    def setUp(self):
        self.set_mlu()
        self.op_type = "elementwise_sub"
        self.init_dtype()
        self.init_input_output()
        self.init_axis()

        self.inputs = {
            'X': OpTest.np_dtype_to_fluid_dtype(self.x),
            'Y': OpTest.np_dtype_to_fluid_dtype(self.y)
        }
        self.attrs = {'axis': self.axis}
        self.outputs = {'Out': self.out}

    def set_mlu(self):
        self.__class__.use_mlu = True
        self.place = paddle.device.MLUPlace(0)

    def init_input_output(self):
        self.x = np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype)
        self.y = np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype)
        self.out = np.subtract(self.x, self.y)

    def init_dtype(self):
        self.dtype = np.float32

    def init_axis(self):
        self.axis = 0

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad_normal(self):
        self.check_grad_with_place(self.place, ['X', 'Y'], 'Out')

    def test_check_grad_ingore_x(self):
        self.check_grad_with_place(self.place, ['Y'],
                                   'Out',
                                   max_relative_error=0.005,
                                   no_grad_set=set("X"))

    def test_check_grad_ingore_y(self):
        self.check_grad_with_place(self.place, ['X'],
                                   'Out',
                                   max_relative_error=0.005,
                                   no_grad_set=set('Y'))


@skip_check_grad_ci(
    reason="[skip shape check] Use y_shape(1) to test broadcast.")
class TestElementwiseSubOp_scalar(TestElementwiseSubOp):

    def setUp(self):
        self.set_mlu()
        self.op_type = "elementwise_sub"
        self.inputs = {
            'X': np.random.rand(10, 3, 4).astype(np.float32),
            'Y': np.random.rand(1).astype(np.float32)
        }
        self.outputs = {'Out': self.inputs['X'] - self.inputs['Y']}


class TestElementwiseSubOp_Vector(TestElementwiseSubOp):

    def setUp(self):
        self.set_mlu()
        self.op_type = "elementwise_sub"
        self.inputs = {
            'X': np.random.random((100, )).astype("float32"),
            'Y': np.random.random((100, )).astype("float32")
        }
        self.outputs = {'Out': self.inputs['X'] - self.inputs['Y']}


class TestElementwiseSubOp_broadcast_0(TestElementwiseSubOp):

    def setUp(self):
        self.set_mlu()
        self.op_type = "elementwise_sub"
        self.inputs = {
            'X': np.random.rand(100, 3, 2).astype(np.float32),
            'Y': np.random.rand(100).astype(np.float32)
        }
        self.attrs = {'axis': 0}
        self.outputs = {
            'Out': self.inputs['X'] - self.inputs['Y'].reshape(100, 1, 1)
        }


class TestElementwiseSubOp_broadcast_1(TestElementwiseSubOp):

    def setUp(self):
        self.set_mlu()
        self.op_type = "elementwise_sub"
        self.inputs = {
            'X': np.random.rand(2, 100, 3).astype(np.float32),
            'Y': np.random.rand(100).astype(np.float32)
        }
        self.attrs = {'axis': 1}
        self.outputs = {
            'Out': self.inputs['X'] - self.inputs['Y'].reshape(1, 100, 1)
        }


class TestElementwiseSubOp_broadcast_2(TestElementwiseSubOp):

    def setUp(self):
        self.set_mlu()
        self.op_type = "elementwise_sub"
        self.inputs = {
            'X': np.random.rand(2, 3, 100).astype(np.float32),
            'Y': np.random.rand(100).astype(np.float32)
        }
        self.outputs = {
            'Out': self.inputs['X'] - self.inputs['Y'].reshape(1, 1, 100)
        }


class TestElementwiseSubOp_broadcast_3(TestElementwiseSubOp):

    def setUp(self):
        self.set_mlu()
        self.op_type = "elementwise_sub"
        self.inputs = {
            'X': np.random.rand(2, 10, 12, 3).astype(np.float32),
            'Y': np.random.rand(10, 12).astype(np.float32)
        }
        self.attrs = {'axis': 1}
        self.outputs = {
            'Out': self.inputs['X'] - self.inputs['Y'].reshape(1, 10, 12, 1)
        }


class TestElementwiseSubOp_broadcast_4(TestElementwiseSubOp):

    def setUp(self):
        self.set_mlu()
        self.op_type = "elementwise_sub"
        self.inputs = {
            'X': np.random.rand(2, 5, 3, 12).astype(np.float32),
            'Y': np.random.rand(2, 5, 1, 12).astype(np.float32)
        }
        self.outputs = {'Out': self.inputs['X'] - self.inputs['Y']}


class TestElementwiseSubOp_commonuse_1(TestElementwiseSubOp):

    def setUp(self):
        self.set_mlu()
        self.op_type = "elementwise_sub"
        self.inputs = {
            'X': np.random.rand(2, 3, 100).astype(np.float32),
            'Y': np.random.rand(1, 1, 100).astype(np.float32)
        }
        self.outputs = {'Out': self.inputs['X'] - self.inputs['Y']}


class TestElementwiseSubOp_commonuse_2(TestElementwiseSubOp):

    def setUp(self):
        self.set_mlu()
        self.op_type = "elementwise_sub"
        self.inputs = {
            'X': np.random.rand(10, 3, 1, 4).astype(np.float32),
            'Y': np.random.rand(10, 1, 12, 1).astype(np.float32)
        }
        self.outputs = {'Out': self.inputs['X'] - self.inputs['Y']}


class TestElementwiseSubOp_xsize_lessthan_ysize(TestElementwiseSubOp):

    def setUp(self):
        self.set_mlu()
        self.op_type = "elementwise_sub"
        self.inputs = {
            'X': np.random.rand(10, 12).astype(np.float32),
            'Y': np.random.rand(2, 3, 10, 12).astype(np.float32)
        }
        self.attrs = {'axis': 2}
        self.outputs = {
            'Out': self.inputs['X'].reshape(1, 1, 10, 12) - self.inputs['Y']
        }


if __name__ == '__main__':
    unittest.main()
