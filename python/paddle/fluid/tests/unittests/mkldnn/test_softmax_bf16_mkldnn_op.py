#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.fluid.tests.unittests.op_test import convert_float_to_uint16
import paddle.fluid.core as core
from paddle.fluid.tests.unittests.test_softmax_op import TestSoftmaxOp, TestSoftmaxOp2, TestSoftmaxOp3, TestSoftmaxOp4, TestSoftmaxOp5, TestSoftmaxOp6
from paddle import enable_static


def stable_softmax(x):
    """Compute the softmax of vector x in a numerically stable way."""
    shiftx = x - np.max(x).clip(-64.)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)


@unittest.skipIf(not core.supports_bfloat16(),
                 "place does not support BF16 evaluation")
class TestSoftmaxMKLDNNOp(TestSoftmaxOp):

    def get_x_shape(self):
        return [10, 10]

    def get_axis(self):
        return -1

    def setUp(self):
        self.op_type = "softmax"
        self.use_mkldnn = True
        self.dtype = np.uint16
        self.init_kernel_type()
        self.shape = self.get_x_shape()
        self.axis = self.get_axis()

        x = np.random.uniform(0.1, 1, self.shape).astype(np.float64)
        out = convert_float_to_uint16(
            np.apply_along_axis(stable_softmax, self.axis, x))

        self.inputs = {'X': convert_float_to_uint16(x)}
        self.outputs = {'Out': out}
        self.attrs = {'axis': self.axis, 'use_mkldnn': self.use_mkldnn}

    def test_check_output(self):
        self.check_output_with_place(core.CPUPlace())

    def test_check_grad(self):
        pass

    def init_kernel_type(self):
        self.use_mkldnn = True


class TestSoftmaxMKLDNNOp2(TestSoftmaxOp2):

    def init_kernel_type(self):
        self.use_mkldnn = True


class TestSoftmaxMKLDNNOp3(TestSoftmaxOp3):

    def init_kernel_type(self):
        self.use_mkldnn = True


class TestSoftmaxMKLDNNOp4(TestSoftmaxOp4):

    def init_kernel_type(self):
        self.use_mkldnn = True


class TestSoftmaxMKLDNNOp5(TestSoftmaxOp5):

    def init_kernel_type(self):
        self.use_mkldnn = True


class TestSoftmaxMKLDNNOp6(TestSoftmaxOp6):

    def init_kernel_type(self):
        self.use_mkldnn = True


if __name__ == '__main__':
    enable_static()
    unittest.main()
