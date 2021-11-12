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

import paddle
import numpy as np
import sys
import unittest
sys.path.append("..")
from op_test_xpu import XPUOpTest
paddle.enable_static()
np.random.seed(10)


def stable_softmax(x):
    """Compute the softmax of vector x in a numerically stable way."""
    # clip to shiftx, otherwise, when calc loss with
    # log(exp(shiftx)), may get log(0)=INF
    shiftx = (x - np.max(x)).clip(-64.)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)


def ref_softmax(x, axis=None, dtype=None):
    x_t = x.copy()
    if dtype is not None:
        x_t = x_t.astype(dtype)
    if axis is None:
        axis = -1
    return np.apply_along_axis(stable_softmax, axis, x_t)


class TestXPUSoftmaxOp(XPUOpTest):
    def setUp(self):
        self.op_type = "softmax"
        self.shape = [2, 3, 4, 5]
        self.axis = -1
        self.set_attrs()
        self.init_type()

        x = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        out = np.apply_along_axis(stable_softmax, self.axis, x)

        self.inputs = {'X': x}
        self.outputs = {'Out': out}
        self.attrs = {'axis': self.axis, 'use_xpu': True}

    def init_type(self):
        self.dtype = np.float16

    def set_attrs(self):
        pass

    def test_check_output(self):
        self.check_output_with_place(paddle.XPUPlace(0), atol=1e-4)

    def test_check_grad(self):
        self.check_grad_with_place(paddle.XPUPlace(0), ['X'], 'Out')


# class TestXPUSoftmaxAxis3(TestXPUSoftmaxOp):
#     def set_attrs(self):
#         self.axis = 3

# class TestXPUSoftmax2D(TestXPUSoftmaxOp):
#     def set_attrs(self):
#         self.shape = [10, 12]

# class TestXPUSoftmax3D(TestXPUSoftmaxOp):
#     def set_attrs(self):
#         self.shape = [4, 5, 6]

# class TestXPUSoftmaxAxis3FP16(TestXPUSoftmaxOp):
#     def set_attrs(self):
#         self.axis = 3
#     def init_type(self):
#         self.dtype = np.float16

# class TestXPUSoftmax2DFP16(TestXPUSoftmaxOp):
#     def set_attrs(self):
#         self.shape = [10, 12]
#     def init_type(self):
#         self.dtype = np.float16

# class TestXPUSoftmax3DFP16(TestXPUSoftmaxOp):
#     def set_attrs(self):
#         self.shape = [4, 5, 6]
#     def init_type(self):
#         self.dtype = np.float16

if __name__ == "__main__":
    unittest.main()
