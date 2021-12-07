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
from op_test import OpTest, _set_use_system_allocator
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core

from paddle.fluid import ParamAttr
from paddle.fluid.framework import Program, grad_var_name
from paddle.fluid.executor import Executor
from paddle.fluid.backward import append_backward

paddle.enable_static()


class TestArgsortOp(OpTest):
    def setUp(self):
        self.set_npu()
        self.op_type = "argsort"
        self.place = paddle.NPUPlace(0)
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

    def set_npu(self):
        self.__class__.use_npu = True
        self.__class__.no_need_check_grad = True

    def init_kernel_type(self):
        self.use_mkldnn = False

    def init_inputshape(self):
        self.input_shape = (2, 2, 2, 3, 3)

    def init_dtype(self):
        self.dtype = np.float16

    def init_axis(self):
        self.axis = -1

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def init_direction(self):
        self.descending = False


class TestArgsortOpAxis0NPU(TestArgsortOp):
    def init_axis(self):
        self.axis = 0


class TestArgsortOpAxis1NPU(TestArgsortOp):
    def init_axis(self):
        self.axis = 1


class TestArgsortOpAxis2NPU(TestArgsortOp):
    def init_axis(self):
        self.axis = 2


class TestArgsortOpAxisNeg1NPU(TestArgsortOp):
    def init_axis(self):
        self.axis = -1


class TestArgsortOpAxisNeg2NPU(TestArgsortOp):
    def init_axis(self):
        self.axis = -2


class TestArgsortOpDescendingAxisNPU(TestArgsortOp):
    def init_direction(self):
        self.descending = True


class TestArgsortOpDescendingAxis0NPU(TestArgsortOpAxis0NPU):
    def init_direction(self):
        self.descending = True


class TestArgsortOpDescendingAxis1NPU(TestArgsortOpAxis1NPU):
    def init_direction(self):
        self.descending = True


class TestArgsortOpDescendingAxis2NPU(TestArgsortOpAxis2NPU):
    def init_direction(self):
        self.descending = True


class TestArgsortOpDescendingAxisNeg1NPU(TestArgsortOpAxisNeg1NPU):
    def init_direction(self):
        self.descending = True


class TestArgsortOpDescendingAxisNeg2NPU(TestArgsortOpAxisNeg2NPU):
    def init_direction(self):
        self.descending = True


# liurui25: argsort of npu has bug with type fp32, 
# it will change the type from fp32 to fp16, 
# so the check_output_with_place add thw atol
# this test is only used to test the grad
# issue： https://gitee.com/ascend/modelzoo/issues/I44I7K


class TestArgsortOpAxis0NPUFP32(TestArgsortOp):
    def init_axis(self):
        self.axis = 0

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-2)

    def set_npu(self):
        self.__class__.use_npu = True

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ["X"], "Out")


class TestArgsortOpAxis1NPUFP32(TestArgsortOpAxis0NPUFP32):
    def init_axis(self):
        self.axis = 1


class TestArgsortOpAxis2NPUFP32(TestArgsortOpAxis0NPUFP32):
    def init_axis(self):
        self.axis = 2


class TestArgsortOpAxisNeg1NPUFP32(TestArgsortOpAxis0NPUFP32):
    def init_axis(self):
        self.axis = -1


class TestArgsortOpAxisNeg2NPUFP32(TestArgsortOpAxis0NPUFP32):
    def init_axis(self):
        self.axis = -2


class TestArgsortOpDescendingAxisNPUFP32(TestArgsortOpAxis0NPUFP32):
    def init_direction(self):
        self.descending = True


class TestArgsortOpDescendingAxis0NPUFP32(TestArgsortOpAxis0NPUFP32):
    def init_direction(self):
        self.descending = True


class TestArgsortOpDescendingAxis1NPUFP32(TestArgsortOpAxis1NPUFP32):
    def init_direction(self):
        self.descending = True


class TestArgsortOpDescendingAxis2NPUFP32(TestArgsortOpAxis2NPUFP32):
    def init_direction(self):
        self.descending = True


class TestArgsortOpDescendingAxisNeg1NPUFP32(TestArgsortOpAxisNeg1NPUFP32):
    def init_direction(self):
        self.descending = True


class TestArgsortOpDescendingAxisNeg2NPUFP32(TestArgsortOpAxisNeg2NPUFP32):
    def init_direction(self):
        self.descending = True


if __name__ == '__main__':
    unittest.main()
