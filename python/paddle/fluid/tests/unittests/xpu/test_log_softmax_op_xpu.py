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
import sys

sys.path.append("..")
from op_test import OpTest

import paddle
import paddle.fluid.core as core
import paddle.nn.functional as F

from op_test_xpu import XPUOpTest
from xpu.get_test_cover_info import create_test_class, get_xpu_op_support_types, XPUOpTestWrapper

paddle.enable_static()
np.random.seed(10)


def ref_log_softmax(x):
    shiftx = (x - np.max(x))
    out = shiftx - np.log(np.exp(shiftx).sum())
    return out


def ref_log_softmax_grad(x, axis):
    if axis < 0:
        axis += len(x.shape)
    out = np.apply_along_axis(ref_log_softmax, axis, x)
    axis_dim = x.shape[axis]
    dout = np.full_like(x, fill_value=1. / x.size)
    dx = dout - np.exp(out) * dout.copy().sum(axis=axis, keepdims=True).repeat(
        axis_dim, axis=axis)
    return dx


class XPUTestLogSoftmaxOp(XPUOpTestWrapper):

    def __init__(self):
        self.op_name = 'log_softmax'
        self.use_dynamic_create_class = True

    def dynamic_create_class(self):
        base_class = self.TestXPULogSoftmaxOp
        classes = []
        axis_arr = [-1, 1]
        shape_arr = [[2, 3, 4, 5], [12, 10], [2, 5], [7, 7], [3, 5, 7]]
        for axis in axis_arr:
            for shape in shape_arr:
                class_name = 'XPUTestLogSoftmax_' + \
                       str(axis) + "_" + str(shape)
                attr_dict = {'axis': axis, 'shape': shape}
                classes.append([class_name, attr_dict])
        return base_class, classes

    class TestXPULogSoftmaxOp(XPUOpTest):

        def setUp(self):
            self.op_type = 'log_softmax'
            self.python_api = F.log_softmax
            self.dtype = 'float32'
            self.set_attrs()
            self.use_xpu = True
            if not hasattr(self, 'axis'):
                self.shape = [2, 3, 4, 5]
                self.axis = -1

            x = np.random.uniform(0.1, 1., self.shape).astype(self.dtype)
            out = np.apply_along_axis(ref_log_softmax, self.axis, x)
            self.x_grad = ref_log_softmax_grad(x, self.axis)

            self.inputs = {'X': x}
            self.outputs = {'Out': out}
            self.attrs = {'axis': self.axis}

        def set_attrs(self):
            pass

        def test_check_output(self):
            self.check_output(check_eager=True)

        def test_check_grad(self):
            self.check_grad(['X'], ['Out'],
                            user_defined_grads=[self.x_grad],
                            check_eager=True)


support_types = get_xpu_op_support_types('log_softmax')
for stype in support_types:
    create_test_class(globals(), XPUTestLogSoftmaxOp, stype)

if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
