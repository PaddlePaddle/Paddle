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
import paddle.fluid.core as core
from op_test import OpTest


class TestSumOp(OpTest):
    def setUp(self):
        self.op_type = "sum"
        self.dtype = np.float32
        self.use_mkldnn = False
        self.init_kernel_type()
        self.init_dtype()

        x0 = np.random.random((3, 4)).astype(self.dtype)
        x1 = np.random.random((3, 4)).astype(self.dtype)
        x2 = np.random.random((3, 4)).astype(self.dtype)
        self.inputs = {"X": [("x0", x0), ("x1", x1), ("x2", x2)]}
        y = x0 + x1 + x2
        self.outputs = {'Out': y}
        self.attrs = {'use_mkldnn': self.use_mkldnn}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['x0'], 'Out')

    def init_kernel_type(self):
        pass

    def init_dtype(self):
        pass


class TestFP16SumOp(TestSumOp):
    def init_dtype(self):
        self.dtype = np.float16

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place):
                self.check_output_with_place(place, atol=1e-3)

    def test_check_grad(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place):
                self.check_grad_with_place(place, ['x0'], 'Out')


if __name__ == "__main__":
    unittest.main()
