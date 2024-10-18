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
from get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)
from op_test import convert_float_to_uint16, skip_check_grad_ci
from op_test_xpu import XPUOpTest

import paddle
import paddle.nn.functional as F

paddle.enable_static()
np.random.seed(10)


def ref_log_softmax(x):
    shiftx = x - np.max(x)
    out = shiftx - np.log(np.exp(shiftx).sum())
    return out


def ref_log_softmax_grad(x, axis):
    if axis < 0:
        axis += len(x.shape)
    out = np.apply_along_axis(ref_log_softmax, axis, x)
    axis_dim = x.shape[axis]
    dout = np.full_like(x, fill_value=1.0 / x.size)
    dx = dout - np.exp(out) * dout.copy().sum(axis=axis, keepdims=True).repeat(
        axis_dim, axis=axis
    )
    return dx


class XPUTestLogSoftmaxOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'log_softmax'
        self.use_dynamic_create_class = False

    class TestXPULogSoftmaxOp(XPUOpTest):
        def setUp(self):
            self.op_type = 'log_softmax'
            self.python_api = F.log_softmax
            self.init_shape()
            if self.in_type == np.uint16:
                self.dtype = np.float32
            else:
                self.dtype = self.in_type
            self.set_attrs()
            self.use_xpu = True
            if not hasattr(self, 'axis'):
                self.shape = [2, 3, 4, 5]
                self.axis = -1

            x = np.random.uniform(0.1, 1.0, self.shape).astype(self.dtype)
            out = np.apply_along_axis(ref_log_softmax, self.axis, x)
            self.x_grad = ref_log_softmax_grad(x, self.axis)

            if self.in_type == np.uint16:
                x = convert_float_to_uint16(x)
                out = convert_float_to_uint16(out)
                self.dtype = self.in_type

            self.inputs = {'X': x}
            self.outputs = {'Out': out}
            self.attrs = {'axis': self.axis}

        def set_attrs(self):
            pass

        def init_shape(self):
            pass

        def test_check_output(self):
            self.check_output(check_dygraph=True)

        def test_check_grad(self):
            self.check_grad(
                ['X'],
                ['Out'],
                user_defined_grads=[self.x_grad],
                check_dygraph=True,
            )

    class TestXPULogSoftmaxOpShape1(TestXPULogSoftmaxOp):
        def init_shape(self):
            self.shape = [2, 3, 4, 5]
            self.axis = 3

    class TestXPULogSoftmaxOpShape2(TestXPULogSoftmaxOp):
        def init_shape(self):
            self.shape = [12, 10]
            self.axis = -1

    class TestXPULogSoftmaxOpShape3(TestXPULogSoftmaxOp):
        def init_shape(self):
            self.shape = [3, 5, 7]
            self.axis = 2

    @skip_check_grad_ci(
        reason="[skip shape check] test log softmax grad when number of input elements less than 100."
    )
    class TestXPULogSoftmaxOp_SmallShape1(TestXPULogSoftmaxOp):
        def init_shape(self):
            self.shape = [2, 5]
            self.axis = 1

    @skip_check_grad_ci(
        reason="[skip shape check] test log softmax grad when number of input elements less than 100."
    )
    class TestXPULogSoftmaxOp_SmallShape2(TestXPULogSoftmaxOp):
        def init_shape(self):
            self.shape = [7, 7]
            self.axis = -1


support_types = get_xpu_op_support_types('log_softmax')
for stype in support_types:
    create_test_class(globals(), XPUTestLogSoftmaxOp, stype)

if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
