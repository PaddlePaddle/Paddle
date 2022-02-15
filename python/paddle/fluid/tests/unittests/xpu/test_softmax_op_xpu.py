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
from xpu.get_test_cover_info import create_test_class, get_xpu_op_support_types, XPUOpTestWrapper

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


class XPUTestSoftmaxOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'softmax'
        self.use_dynamic_create_class = True

    def dynamic_create_class(self):
        base_class = self.TestSoftmaxOp
        classes = []
        shapes = [[2, 3, 4, 5], [7, 1], [63, 18], [2, 38512], [3, 4095]]
        axis = [-1, 0, 1]
        for shape in shapes:
            for axi in axis:
                class_name = 'XPUTestSoftmax_' + \
                       str(shape) + "_" + str(axi)
                attr_dict = {'shape': shape, 'axis': axi}
                classes.append([class_name, attr_dict])
        return base_class, classes

    class TestSoftmaxOp(XPUOpTest):
        def setUp(self):
            self.op_type = "softmax"
            if not hasattr(self, 'shape'):
                self.shape = [1, 7]
                self.axis = -1
            self.dtype = np.float32

            x = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
            out = np.apply_along_axis(stable_softmax, self.axis, x)

            self.inputs = {'X': x}
            self.outputs = {'Out': out}
            self.attrs = {'axis': self.axis, 'use_xpu': True}

        def test_check_output(self):
            self.check_output_with_place(paddle.XPUPlace(0), atol=1e-4)

        def test_check_grad(self):
            self.check_grad_with_place(paddle.XPUPlace(0), ['X'], 'Out')


support_types = get_xpu_op_support_types('softmax')
for stype in support_types:
    create_test_class(globals(), XPUTestSoftmaxOp, stype)

if __name__ == "__main__":
    unittest.main()
