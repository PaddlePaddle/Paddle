#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import unittest
import numpy as np
from op_test import OpTest
import paddle.fluid.core as core


class TestArgsortOp(OpTest):
    def setUp(self):
        self.init_axis()
        self.init_datatype()
        self.init_direction()
        x = np.random.random((2, 3, 4, 5, 10)).astype(self.dtype)
        self.attrs = {'axis': self.axis, 'descending': self.descending}
        if self.axis < 0:
            self.axis = self.axis + len(x.shape)
        if self.descending:
            self.indices = np.flip(
                np.argsort(
                    x, kind='quicksort', axis=self.axis), self.axis)
            self.out = np.flip(
                np.sort(
                    x, kind='quicksort', axis=self.axis), self.axis)
        else:
            self.indices = np.argsort(x, kind='quicksort', axis=self.axis)
            self.out = np.sort(x, kind='quicksort', axis=self.axis)

        self.op_type = "argsort"
        self.inputs = {'X': x}
        self.outputs = {'Indices': self.indices, 'Out': self.out}

    def init_axis(self):
        self.axis = -1

    def init_datatype(self):
        self.dtype = "float64"

    def init_direction(self):
        self.descending = False

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestArgsortOpAxis0(TestArgsortOp):
    def init_axis(self):
        self.axis = 0


class TestArgsortOpAxis1(TestArgsortOp):
    def init_axis(self):
        self.axis = 1


class TestArgsortOpAxis2(TestArgsortOp):
    def init_axis(self):
        self.axis = 2


class TestArgsortOpAxisNeg1(TestArgsortOp):
    def init_axis(self):
        self.axis = -1


class TestArgsortOpAxisNeg2(TestArgsortOp):
    def init_axis(self):
        self.axis = -2


class TestArgsortOpFP16(TestArgsortOp):
    def init_datatype(self):
        if core.is_compiled_with_cuda():
            self.dtype = 'float16'

    def test_check_output(self):
        pass

    def test_check_output_with_place(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            self.check_output_with_place(place, atol=1e-5)


class TestArgsortOpFP16Axis0(TestArgsortOpFP16):
    def init_axis(self):
        self.axis = 0


class TestArgsortOpFP16Axis2(TestArgsortOpFP16):
    def init_axis(self):
        self.axis = 2


class TestArgsortOpFP16AxisNeg2(TestArgsortOpFP16):
    def init_axis(self):
        self.axis = -2


class TestArgsortOpFP16Axis4Neg4(TestArgsortOpFP16):
    def init_axis(self):
        self.axis = -4


class TestArgsortOpDescendingAxis(TestArgsortOp):
    def init_direction(self):
        self.descending = True


class TestArgsortOpDescendingAxis0(TestArgsortOpAxis0):
    def init_direction(self):
        self.descending = True


class TestArgsortOpDescendingAxis1(TestArgsortOpAxis1):
    def init_direction(self):
        self.descending = True


class TestArgsortOpDescendingAxis2(TestArgsortOpAxis2):
    def init_direction(self):
        self.descending = True


class TestArgsortOpDescendingAxisNeg1(TestArgsortOpAxisNeg1):
    def init_direction(self):
        self.descending = True


class TestArgsortOpDescendingAxisNeg2(TestArgsortOpAxisNeg2):
    def init_direction(self):
        self.descending = True


class TestArgsortOpFP32Axis(TestArgsortOp):
    def init_datatype(self):
        self.dtype = "float32"


class TestArgsortOpFP32DescendingAxis(TestArgsortOp):
    def init_datatype(self):
        self.dtype = "float32"

    def init_direction(self):
        self.descending = True


if __name__ == "__main__":
    unittest.main()
