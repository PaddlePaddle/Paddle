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

import unittest
import numpy as np
import sys

sys.path.append("..")

import paddle
import paddle.fluid as fluid
from paddle.fluid import core
from paddle.fluid import compiler, Program, program_guard

import op_test
from op_test import OpTest, skip_check_grad_ci
from op_test_xpu import XPUOpTest
from xpu.get_test_cover_info import create_test_class, get_xpu_op_support_types, XPUOpTestWrapper

paddle.enable_static()


def huber_loss_forward(val, delta):
    abs_val = abs(val)
    if abs_val <= delta:
        return 0.5 * val * val
    else:
        return delta * (abs_val - 0.5 * delta)


# 1.动态生成不同参数的测试case，wrapper类中必须实现dynamic_create_class方法
# self.use_dynamic_create_class置为True
class XPUTestArgsortOp1(XPUOpTestWrapper):

    def __init__(self):
        self.op_name = 'argsort'
        self.use_dynamic_create_class = True

    def dynamic_create_class(self):
        base_class = self.TestArgsortOp
        classes = []
        for descending in [True, False]:
            for axis in [0, 1, 2, -1, -2]:
                class_name = 'XPUTestArgsortOp_axis_' + str(axis) + '_' + str(
                    descending)
                attr_dict = {'init_axis': axis, 'init_descending': descending}
                classes.append([class_name, attr_dict])
        return base_class, classes

    class TestArgsortOp(XPUOpTest):

        def setUp(self):
            self.op_type = "argsort"
            self.place = paddle.XPUPlace(0)
            self.__class__.no_need_check_grad = True
            self.dtype = self.in_type

            self.input_shape = (2, 2, 2, 3, 3)
            self.axis = -1 if not hasattr(self, 'init_axis') else self.init_axis
            self.descending = False if not hasattr(
                self, 'init_descending') else self.init_descending

            if self.in_type == np.float32:
                self.x = np.random.random(self.input_shape).astype(self.dtype)
            else:
                self.x = np.random.randint(low=-1000,
                                           high=1000,
                                           size=self.input_shape).astype(
                                               self.dtype)
            self.inputs = {"X": self.x}
            self.attrs = {"axis": self.axis, "descending": self.descending}
            self.get_output()
            self.outputs = {"Out": self.sorted_x, "Indices": self.indices}

        def get_output(self):
            if self.descending:
                self.indices = np.flip(
                    np.argsort(self.x, kind='heapsort', axis=self.axis),
                    self.axis)
                self.sorted_x = np.flip(
                    np.sort(self.x, kind='heapsort', axis=self.axis), self.axis)
            else:
                self.indices = np.argsort(self.x,
                                          kind='heapsort',
                                          axis=self.axis)
                self.sorted_x = np.sort(self.x, kind='heapsort', axis=self.axis)

        def test_check_output(self):
            self.check_output_with_place(self.place)


# 2. 为不同参数的测试case定义一个测试类，self.use_dynamic_create_class需要置为False
class XPUTestArgsortOp2(XPUOpTestWrapper):

    def __init__(self):
        self.op_name = 'argsort'
        self.use_dynamic_create_class = False

    class TestArgsortOp(XPUOpTest):

        def setUp(self):
            self.op_type = "argsort"
            self.place = paddle.XPUPlace(0)
            self.__class__.no_need_check_grad = True

            self.init_dtype()
            self.init_inputshape()
            self.init_axis()
            self.init_direction()

            if self.in_type == np.float32:
                self.x = np.random.random(self.input_shape).astype(self.dtype)
            else:
                self.x = np.random.randint(low=-1000,
                                           high=1000,
                                           size=self.input_shape).astype(
                                               self.dtype)
            self.inputs = {"X": self.x}
            self.attrs = {"axis": self.axis, "descending": self.descending}
            self.get_output()
            self.outputs = {"Out": self.sorted_x, "Indices": self.indices}

        def get_output(self):
            if self.descending:
                self.indices = np.flip(
                    np.argsort(self.x, kind='heapsort', axis=self.axis),
                    self.axis)
                self.sorted_x = np.flip(
                    np.sort(self.x, kind='heapsort', axis=self.axis), self.axis)
            else:
                self.indices = np.argsort(self.x,
                                          kind='heapsort',
                                          axis=self.axis)
                self.sorted_x = np.sort(self.x, kind='heapsort', axis=self.axis)

        def init_inputshape(self):
            self.input_shape = (2, 2, 2, 3, 3)

        def init_dtype(self):
            self.dtype = self.in_type

        def init_axis(self):
            self.axis = -1

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def init_direction(self):
            self.descending = False

    class TestArgsortOpAxis0XPU(TestArgsortOp):

        def init_axis(self):
            self.axis = 0

    class TestArgsortOpAxis1XPU(TestArgsortOp):

        def init_axis(self):
            self.axis = 1

    class TestArgsortOpAxis2XPU(TestArgsortOp):

        def init_axis(self):
            self.axis = 2

    class TestArgsortOpAxisNeg1XPU(TestArgsortOp):

        def init_axis(self):
            self.axis = -1

    class TestArgsortOpAxisNeg2XPU(TestArgsortOp):

        def init_axis(self):
            self.axis = -2

    class TestArgsortOpDescendingAxisXPU(TestArgsortOp):

        def init_direction(self):
            self.descending = True

    class TestArgsortOpDescendingAxis0XPU(TestArgsortOpAxis0XPU):

        def init_direction(self):
            self.descending = True

    class TestArgsortOpDescendingAxis1XPU(TestArgsortOpAxis1XPU):

        def init_direction(self):
            self.descending = True

    class TestArgsortOpDescendingAxis2XPU(TestArgsortOpAxis2XPU):

        def init_direction(self):
            self.descending = True

    class TestArgsortOpDescendingAxisNeg1XPU(TestArgsortOpAxisNeg1XPU):

        def init_direction(self):
            self.descending = True

    class TestArgsortOpDescendingAxisNeg2XPU(TestArgsortOpAxisNeg2XPU):

        def init_direction(self):
            self.descending = True


support_types = get_xpu_op_support_types('argsort')
for stype in support_types:
    create_test_class(globals(), XPUTestArgsortOp1, stype)
    create_test_class(globals(), XPUTestArgsortOp2, stype)


class XPUTestHuberLossOp(XPUOpTestWrapper):

    def __init__(self):
        self.op_name = 'huber_loss'
        self.use_dynamic_create_class = False

    class TestHuberLossOp(XPUOpTest):

        def setUp(self):
            self.op_type = 'huber_loss'
            self.place = paddle.XPUPlace(0)
            self.dtype = self.in_type

            self.set_inputs()
            self.set_attrs()
            self.set_outputs()

        def set_inputs(self):
            shape = self.set_shape()
            x = np.random.uniform(0, 1., shape).astype(self.dtype)
            y = np.random.uniform(0, 1., shape).astype(self.dtype)
            self.inputs = {
                'X': OpTest.np_dtype_to_fluid_dtype(x),
                'Y': OpTest.np_dtype_to_fluid_dtype(y)
            }

        def set_attrs(self):
            self.attrs = {'delta': 0.5}

        def set_outputs(self):
            delta = self.attrs['delta']
            shape = self.set_shape()
            residual = self.inputs['Y'] - self.inputs['X']
            loss = np.vectorize(huber_loss_forward)(residual,
                                                    delta).astype(self.dtype)
            self.outputs = {'Residual': residual, 'Out': loss.reshape(shape)}

        def set_shape(self):
            return (100, 1)

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def test_check_grad_normal(self):
            self.check_grad_with_place(self.place, ['X', 'Y'], 'Out')

        def test_check_grad_ingore_x(self):
            self.check_grad_with_place(self.place, ['Y'],
                                       'Out',
                                       no_grad_set=set("residual"))

        def test_check_grad_ingore_y(self):
            self.check_grad_with_place(self.place, ['X'],
                                       'Out',
                                       no_grad_set=set('residual'))

    class TestHuberLossOp1(TestHuberLossOp):

        def set_shape(self):
            return (640)

    class TestHuberLossOp2(TestHuberLossOp):

        def set_shape(self):
            return (10, 10)

    class TestHuberLossOp3(TestHuberLossOp):

        def set_shape(self):
            return (10, 10, 1)


support_types = get_xpu_op_support_types('huber_loss')
for stype in support_types:
    create_test_class(globals(), XPUTestHuberLossOp, stype)
    create_test_class(globals(),
                      XPUTestHuberLossOp,
                      stype,
                      ignore_device_version=[core.XPUVersion.XPU1])

if __name__ == '__main__':
    unittest.main()
