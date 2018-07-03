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

import unittest
import numpy as np
from op_test import OpTest
import paddle.fluid.core as core


def stable_softmax(x):
    """Compute the softmax of vector x in a numerically stable way."""
    shiftx = x - np.max(x).clip(-64.)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)


class TestSoftmaxOp(OpTest):
    def setUp(self):
        self.op_type = "softmax"
        self.use_cudnn = False
        self.use_mkldnn = False
        self.dtype = np.float32
        self.init_kernel_type()

        x = np.random.uniform(0.1, 1, [10, 10]).astype(self.dtype)
        out = np.apply_along_axis(stable_softmax, 1, x)
        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}
        self.attrs = {
            'use_cudnn': self.use_cudnn,
            'use_mkldnn': self.use_mkldnn
        }

    def init_kernel_type(self):
        pass

    def test_check_output(self):
        if self.use_cudnn:
            place = core.CUDAPlace(0)
            self.check_output_with_place(place, atol=1e-5)
        else:
            self.check_output()

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        if self.use_cudnn:
            place = core.CUDAPlace(0)
            self.check_grad_with_place(
                place, ["X"], "Out", max_relative_error=0.01)
        else:
            self.check_grad(["X"], "Out", max_relative_error=0.01)


class TestSoftmaxCUDNNOp(TestSoftmaxOp):
    def init_kernel_type(self):
        self.use_cudnn = True


class TestSoftmaxFP16Op(TestSoftmaxOp):
    def init_kernel_type(self):
        self.dtype = np.float16

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place):
                self.check_output_with_place(place, atol=1e-3)


class TestSoftmaxFP16CUDNNOp(TestSoftmaxOp):
    def init_kernel_type(self):
        self.use_cudnn = True
        self.dtype = np.float16

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place):
                self.check_output_with_place(place, atol=1e-3)


class TestSoftmaxMKLDNNOp(TestSoftmaxOp):
    def init_kernel_type(self):
        self.use_mkldnn = True


if __name__ == "__main__":
    unittest.main()
