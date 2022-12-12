#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import paddle.fluid.core as core
from paddle import enable_static
from paddle.fluid.tests.unittests.test_softmax_op import TestSoftmaxOp

enable_static()


def stable_softmax(x):
    """Compute the softmax of vector x in a numerically stable way."""
    shiftx = x - np.max(x).clip(-64.0)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)


class TestSoftmaxMKLDNNOpInt8(TestSoftmaxOp):
    def get_x_shape(self):
        return [10, 10]

    def get_axis(self):
        return -1

    def setUp(self):
        self.op_type = "softmax"
        self.use_mkldnn = True
        self.dtype = np.int8
        self.init_kernel_type()
        self.shape = self.get_x_shape()
        self.axis = self.get_axis()

        x = np.random.uniform(-1, 1, self.shape).astype(np.float64)
        out = np.apply_along_axis(stable_softmax, self.axis, x)

        x_int8 = np.round(x * 128 / np.max(np.abs(x))).astype("int8")

        scale_out = 255 / np.max(np.abs(out))
        out_uint8 = np.round(out * scale_out).astype("uint8")

        self.inputs = {'X': x_int8}
        self.outputs = {'Out': out_uint8}
        self.attrs = {
            'axis': self.axis,
            'use_mkldnn': self.use_mkldnn,
            'Scale_out': scale_out,
            'mkldnn_data_type': "int8",
        }

    def test_check_output(self):
        self.check_output_with_place(core.CPUPlace())

    def test_check_grad(self):
        pass

    def init_kernel_type(self):
        self.use_mkldnn = True


if __name__ == '__main__':
    unittest.main()
