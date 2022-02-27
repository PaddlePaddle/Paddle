#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
import numpy as np
import paddle
from paddle.fluid import core
from paddle.fluid.tests.unittests.test_log_softmax import ref_log_softmax
from paddle.fluid.tests.unittests.op_test import OpTest, OpTestTool, convert_float_to_uint16


@OpTestTool.skip_if_not_cpu_bf16()
class TestLogSoftmaxOneDNNOp(OpTest):
    def setUp(self):
        self.op_type = 'log_softmax'
        self.set_dtype()
        self.set_shape()
        self.set_axis()

        x = np.random.uniform(0.1, 1.0, self.shape).astype(np.float32)
        out = np.apply_along_axis(ref_log_softmax, self.axis, x)

        if self.dtype == np.uint16:
            x = convert_float_to_uint16(x)

        self.inputs = {'X': x}
        self.outputs = {'Out': out}
        self.attrs = {'axis': self.axis, 'use_mkldnn': True}

    def set_dtype(self):
        self.dtype = np.float32

    def set_shape(self):
        self.shape = [2, 3, 4, 5]

    def set_axis(self):
        self.axis = -1

    def test_check_output(self):
        self.check_output_with_place(core.CPUPlace())


class TestLogSoftmax1DOneDNNOp(TestLogSoftmaxOneDNNOp):
    def set_shape(self):
        self.shape = [100]


class TestLogSoftmax3DOneDNNOp(TestLogSoftmaxOneDNNOp):
    def set_shape(self):
        self.shape = [12, 10, 3]


class TestLogSoftmax5DOneDNNOp(TestLogSoftmaxOneDNNOp):
    def set_shape(self):
        self.shape = [2, 3, 4, 5, 6]


class TestLogSoftmaxPositiveAxisOneDNNOp(TestLogSoftmaxOneDNNOp):
    def set_axis(self):
        self.axis = 2


# BF16 TESTS
class TestLogSoftmax1DBF16OneDNNOp(TestLogSoftmax1DOneDNNOp):
    def set_dtype(self):
        self.dtype = np.uint16


class TestLogSoftmaxPositiveAxisBF16OneDNNOp(
        TestLogSoftmaxPositiveAxisOneDNNOp):
    def set_dtype(self):
        self.dtype = np.uint16


class TestLogSoftmax5DBF16OneDNNOp(TestLogSoftmax5DOneDNNOp):
    def set_shape(self):
        self.shape = [2, 3, 4, 5, 6]


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
