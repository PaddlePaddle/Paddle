# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import numpy as np
import unittest
import sys
sys.path.append("..")
from op_test import OpTest
import paddle
import paddle.fluid as fluid
from paddle.fluid import core
import paddle.nn.functional as F
from test_log_softmax import ref_log_softmax, ref_log_softmax_grad
paddle.enable_static()
np.random.seed(10)


class TestLogSoftmaxNPUOp(OpTest):
    def setUp(self):
        self.set_npu()
        self.place = paddle.NPUPlace(0)
        self.op_type = "log_softmax"
        self.dtype = np.float32
        self.shape = [2, 3, 4, 5]
        self.axis = -1
        self.set_attrs()
        self.set_dtype()
        x = np.random.uniform(0.1, 1., self.shape).astype(self.dtype)
        out = np.apply_along_axis(ref_log_softmax, self.axis, x)
        self.x_grad = ref_log_softmax_grad(x, self.axis)
        self.inputs = {'X': x}
        self.outputs = {'Out': out}
        self.attrs = {'axis': self.axis}

    def set_npu(self):
        self.__class__.use_npu = True
        self.__class__.no_need_check_grad = True

    def set_attrs(self):
        pass

    def set_dtype(self):
        pass

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        pass


def test_class(op_type, typename):
    class TestLogSoftmaxShape(TestLogSoftmaxNPUOp):
        def set_attrs(self):
            self.shape = [12, 10]

        def set_dtype(self):
            self.dtype = typename

    cls_name = "{0}_{1}_1".format(op_type, typename)
    TestLogSoftmaxShape.__name__ = cls_name
    globals()[cls_name] = TestLogSoftmaxShape


def test_class2(op_type, typename):
    class TestLogSoftmaxAxis(TestLogSoftmaxNPUOp):
        def set_attrs(self):
            self.axis = 0

        def set_dtype(self):
            self.dtype = typename

    cls_name = "{0}_{1}_2".format(op_type, typename)

    TestLogSoftmaxAxis.__name__ = cls_name
    globals()[cls_name] = TestLogSoftmaxAxis


for _typename in {'float32'}:
    test_class("logsoftmax", _typename)
    test_class2("logsoftmax", _typename)
if __name__ == '__main__':
    unittest.main()
