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

import paddle.fluid as fluid
import paddle

import numpy as np
import unittest
import sys

sys.path.append("..")
from op_test import OpTest

paddle.enable_static()
SEED = 2022


def ComputeGrad(x, y, out, axis):
    grad = 1 / out.size
    shape_x = x.shape
    shape_y = y.shape
    shape_out = out.shape
    reduce_axes_x = []
    reduce_axes_y = []

    if shape_x != shape_out:
        if len(shape_x) < len(shape_out):
            src_axis = axis
        else:
            src_axis = 0

        for ax in range(len(shape_out)):
            if (ax < src_axis or ax >= src_axis + len(shape_x)) or (
                    shape_out[ax] > 1 and shape_x[ax - src_axis] == 1):
                reduce_axes_x.append(ax)

    if shape_y != shape_out:
        if len(shape_y) < len(shape_out):
            src_axis = axis
        else:
            src_axis = 0

        for ax in range(len(shape_out)):
            if (ax < src_axis or ax >= src_axis + len(shape_y)) or (
                    shape_out[ax] > 1 and shape_y[ax - src_axis] == 1):
                reduce_axes_y.append(ax)

    if len(reduce_axes_x) > 0:
        for i in reduce_axes_x:
            x = np.expand_dims(x, axis=i)

    if len(reduce_axes_y) > 0:
        for i in reduce_axes_y:
            y = np.expand_dims(y, axis=i)

    dx = y * np.power(x, y - 1) * grad
    dy = np.log(x) * np.power(x, y) * grad

    if len(reduce_axes_x) > 0:
        for i, element in enumerate(reduce_axes_x):
            dx = np.add.reduce(dx, element - i)

    if len(reduce_axes_y) > 0:
        for i, element in enumerate(reduce_axes_y):
            dy = np.add.reduce(dy, element - i)

    return dx, dy


class TestElementwisePow(OpTest):

    def setUp(self):
        self.set_mlu()
        self.op_type = "elementwise_pow"

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

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def init_axis(self):
        self.axis = -1

    def init_input_output(self):
        np.random.seed(SEED)
        self.x = np.random.uniform(1, 2, [11, 17]).astype(self.dtype)
        self.y = np.random.uniform(1, 2, [11, 17]).astype(self.dtype)
        self.out = np.power(self.x, self.y)

    def test_check_grad_normal(self):
        dx, dy = ComputeGrad(self.x, self.y, self.out, self.axis)
        self.check_grad_with_place(self.place, ['X', 'Y'],
                                   'Out',
                                   user_defined_grads=[dx, dy])

    def test_check_grad_ingore_x(self):
        _, dy = ComputeGrad(self.x, self.y, self.out, self.axis)
        self.check_grad_with_place(self.place, ['Y'],
                                   'Out',
                                   no_grad_set=set("X"),
                                   user_defined_grads=[dy])

    def test_check_grad_ingore_y(self):
        dx, _ = ComputeGrad(self.x, self.y, self.out, self.axis)
        self.check_grad_with_place(self.place, ['X'],
                                   'Out',
                                   no_grad_set=set("Y"),
                                   user_defined_grads=[dx])


class TestElementwisePowFp16(TestElementwisePow):

    def init_input_output(self):
        np.random.seed(SEED)
        self.x = np.random.uniform(1, 2, [11, 17]).astype(self.dtype)
        self.y = np.random.uniform(1, 2, [11, 17]).astype(self.dtype)
        self.out = np.power(self.x, self.y)

    def set_mlu(self):
        self.__class__.use_mlu = True
        # self.__class__.no_need_check_grad = True
        self.place = paddle.device.MLUPlace(0)

    def init_dtype(self):
        self.dtype = np.float16

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-5)


class TestElementwisePowOp_broadcast_0(TestElementwisePow):

    def init_axis(self):
        self.axis = 1

    def init_input_output(self):
        np.random.seed(SEED)
        self.x = np.random.uniform(1, 2, [11, 17]).astype(self.dtype)
        self.y = np.random.uniform(1, 2, [1, 11, 17]).astype(self.dtype)
        self.out = np.power(self.x, self.y)

    def test_check_grad_normal(self):
        dx, dy = ComputeGrad(self.x, self.y, self.out, self.axis)
        self.check_grad_with_place(self.place, ['X', 'Y'],
                                   'Out',
                                   user_defined_grads=[dx, dy])

    def test_check_grad_ingore_x(self):
        _, dy = ComputeGrad(self.x, self.y, self.out, self.axis)
        self.check_grad_with_place(self.place, ['Y'],
                                   'Out',
                                   no_grad_set=set("X"),
                                   user_defined_grads=[dy])

    def test_check_grad_ingore_y(self):
        dx, _ = ComputeGrad(self.x, self.y, self.out, self.axis)
        self.check_grad_with_place(self.place, ['X'],
                                   'Out',
                                   no_grad_set=set("Y"),
                                   user_defined_grads=[dx])


class TestElementwisePowOp_broadcast_1(TestElementwisePow):

    def init_axis(self):
        self.axis = 1

    def init_input_output(self):
        np.random.seed(SEED)
        self.x = np.random.uniform(1, 2, [2, 100, 1]).astype(self.dtype)
        self.y = np.random.uniform(1, 2, [100]).astype(self.dtype)
        self.out = np.power(self.x, self.y.reshape(1, 100, 1))

    def test_check_grad_normal(self):
        dx, dy = ComputeGrad(self.x, self.y, self.out, self.axis)
        self.check_grad_with_place(self.place, ['X', 'Y'],
                                   'Out',
                                   user_defined_grads=[dx, dy])

    def test_check_grad_ingore_x(self):
        _, dy = ComputeGrad(self.x, self.y, self.out, self.axis)
        self.check_grad_with_place(self.place, ['Y'],
                                   'Out',
                                   no_grad_set=set("X"),
                                   user_defined_grads=[dy])

    def test_check_grad_ingore_y(self):
        dx, _ = ComputeGrad(self.x, self.y, self.out, self.axis)
        self.check_grad_with_place(self.place, ['X'],
                                   'Out',
                                   no_grad_set=set("Y"),
                                   user_defined_grads=[dx])


class TestElementwisePowOp_broadcast_2(TestElementwisePow):

    def init_axis(self):
        self.axis = 0

    def init_input_output(self):
        np.random.seed(SEED)
        self.x = np.random.uniform(0.1, 1, [100, 3, 1]).astype(self.dtype)
        self.y = np.random.uniform(0.1, 1, [100]).astype(self.dtype)
        self.out = np.power(self.x, self.y.reshape(100, 1, 1))

    def test_check_grad_normal(self):
        dx, dy = ComputeGrad(self.x, self.y, self.out, self.axis)
        self.check_grad_with_place(self.place, ['X', 'Y'],
                                   'Out',
                                   user_defined_grads=[dx, dy])

    def test_check_grad_ingore_x(self):
        _, dy = ComputeGrad(self.x, self.y, self.out, self.axis)
        self.check_grad_with_place(self.place, ['Y'],
                                   'Out',
                                   no_grad_set=set("X"),
                                   user_defined_grads=[dy])

    def test_check_grad_ingore_y(self):
        dx, _ = ComputeGrad(self.x, self.y, self.out, self.axis)
        self.check_grad_with_place(self.place, ['X'],
                                   'Out',
                                   no_grad_set=set("Y"),
                                   user_defined_grads=[dx])


if __name__ == '__main__':
    unittest.main()
