#  Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest
from op_test_xpu import XPUOpTest
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core

from paddle.fluid import ParamAttr
from paddle.fluid.framework import Program, grad_var_name
from paddle.fluid.executor import Executor
from paddle.fluid.backward import append_backward

paddle.enable_static()


class TestArgsortOp(XPUOpTest):
    def setUp(self):
        self.set_xpu()
        self.op_type = "argsort"
        self.place = paddle.XPUPlace(0)
        self.init_dtype()
        self.init_inputshape()
        self.init_axis()
        self.init_direction()

        self.x = np.random.random(self.input_shape).astype(self.dtype)
        self.inputs = {"X": self.x}
        self.attrs = {"axis": self.axis, "descending": self.descending}
        self.get_output()
        self.outputs = {"Out": self.sorted_x, "Indices": self.indices}

    def get_output(self):
        if self.descending:
            self.indices = np.flip(
                np.argsort(
                    self.x, kind='heapsort', axis=self.axis), self.axis)
            self.sorted_x = np.flip(
                np.sort(
                    self.x, kind='heapsort', axis=self.axis), self.axis)
        else:
            self.indices = np.argsort(self.x, kind='heapsort', axis=self.axis)
            self.sorted_x = np.sort(self.x, kind='heapsort', axis=self.axis)

    def set_xpu(self):
        self.__class__.use_xpu = True
        self.__class__.no_need_check_grad = True

    def init_inputshape(self):
        self.input_shape = (2, 2, 2, 3, 3)

    def init_dtype(self):
        self.dtype = 'float32'

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


class TestArgsortOpAxis0XPUINT64(TestArgsortOp):
    def setUp(self):
        self.set_xpu()
        self.op_type = "argsort"
        self.place = paddle.XPUPlace(0)
        self.init_dtype()
        self.init_inputshape()
        self.init_axis()
        self.init_direction()

        self.x = np.random.randint(
            low=-1000, high=1000, size=self.input_shape).astype(self.dtype)
        self.inputs = {"X": self.x}
        self.attrs = {"axis": self.axis, "descending": self.descending}
        self.get_output()
        self.outputs = {"Out": self.sorted_x, "Indices": self.indices}

    def init_axis(self):
        self.axis = 0

    def init_dtype(self):
        self.dtype = 'int64'


class TestArgsortOpAxis1XPUINT64(TestArgsortOpAxis0XPUINT64):
    def init_axis(self):
        self.axis = 1


class TestArgsortOpAxis2XPUINT64(TestArgsortOpAxis0XPUINT64):
    def init_axis(self):
        self.axis = 2


class TestArgsortOpAxisNeg1XPUINT64(TestArgsortOpAxis0XPUINT64):
    def init_axis(self):
        self.axis = -1


class TestArgsortOpAxisNeg2XPUINT64(TestArgsortOpAxis0XPUINT64):
    def init_axis(self):
        self.axis = -2


class TestArgsortOpDescendingAxisXPUINT64(TestArgsortOpAxis0XPUINT64):
    def init_direction(self):
        self.descending = True


class TestArgsortOpDescendingAxis0XPUINT64(TestArgsortOpAxis0XPUINT64):
    def init_direction(self):
        self.descending = True


class TestArgsortOpDescendingAxis1XPUINT64(TestArgsortOpAxis1XPUINT64):
    def init_direction(self):
        self.descending = True


class TestArgsortOpDescendingAxis2XPUINT64(TestArgsortOpAxis2XPUINT64):
    def init_direction(self):
        self.descending = True


class TestArgsortOpDescendingAxisNeg1XPUINT64(TestArgsortOpAxisNeg1XPUINT64):
    def init_direction(self):
        self.descending = True


class TestArgsortOpDescendingAxisNeg2XPUINT64(TestArgsortOpAxisNeg2XPUINT64):
    def init_direction(self):
        self.descending = True


class TestArgsortOpAxis0XPUINT(TestArgsortOp):
    def setUp(self):
        self.set_xpu()
        self.op_type = "argsort"
        self.place = paddle.XPUPlace(0)
        self.init_dtype()
        self.init_inputshape()
        self.init_axis()
        self.init_direction()

        self.x = np.random.randint(
            low=-1000, high=1000, size=self.input_shape).astype(self.dtype)
        self.inputs = {"X": self.x}
        self.attrs = {"axis": self.axis, "descending": self.descending}
        self.get_output()
        self.outputs = {"Out": self.sorted_x, "Indices": self.indices}

    def init_axis(self):
        self.axis = 0

    def init_dtype(self):
        self.dtype = 'int'


if __name__ == '__main__':
    unittest.main()
