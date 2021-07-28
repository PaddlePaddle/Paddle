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
import paddle
import paddle.fluid as fluid
from paddle.fluid import core

paddle.enable_static()
np.random.seed(10)


def ref_log_softmax(x):
    shiftx = (x - np.max(x))
    out = shiftx - np.log(np.exp(shiftx).sum())
    return out


class TestLogSoftmaxNPUOp(OpTest):
    def setUp(self):
        self.set_npu()
        self.place = paddle.NPUPlace(0)
        self.op_type = "log_softmax"

        self.dtype = np.float32
        self.shape = [2, 3, 4, 5]
        self.axis = -1

        self.set_attrs()

        x = np.random.uniform(0.1, 1., self.shape).astype(self.dtype)
        out = np.apply_along_axis(ref_log_softmax, self.axis, x)

        self.inputs = {'X': x}
        self.outputs = {'Out': out}
        self.attrs = {'axis': self.axis}

    def set_npu(self):
        self.__class__.use_npu = True

    def set_attrs(self):
        pass

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestLogSoftmaxShape(TestLogSoftmaxNPUOp):
    def set_attrs(self):
        self.shape = [12, 10]


# class TestLogSoftmaxAxis(TestLogSoftmaxNPUOp):
#     def set_attrs(self):
#         self.axis = 1


class TestLogSoftmaxFloat64(TestLogSoftmaxNPUOp):
    def set_attrs(self):
        self.dtype = np.float64


class TestLogSoftmaxFloat16(TestLogSoftmaxNPUOp):
    def set_npu(self):
        self.__class__.use_npu = True
        self.__class__.no_need_check_grad = True

    def set_attrs(self):
        self.dtype = np.float16

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-3)


if __name__ == '__main__':
    unittest.main()
