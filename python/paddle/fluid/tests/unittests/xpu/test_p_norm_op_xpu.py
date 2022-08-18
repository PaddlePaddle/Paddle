# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import numpy as np
import sys
import unittest
from functools import reduce

sys.path.append("..")
from op_test import OpTest
from op_test_xpu import XPUOpTest
from operator import mul
from xpu.get_test_cover_info import create_test_class, get_xpu_op_support_types, XPUOpTestWrapper

paddle.enable_static()


def ref_p_norm(x, axis, porder, keepdims=False, reduce_all=False):
    r = []
    if axis is None or reduce_all:
        x = x.flatten()
        if porder == np.inf:
            r = np.amax(np.abs(x), keepdims=keepdims)
        elif porder == -np.inf:
            r = np.amin(np.abs(x), keepdims=keepdims)
        else:
            r = np.linalg.norm(x, ord=porder, keepdims=keepdims)
    elif isinstance(axis, list or tuple) and len(axis) == 2:
        if porder == np.inf:
            axis = tuple(axis)
            r = np.amax(np.abs(x), axis=axis, keepdims=keepdims)
        elif porder == -np.inf:
            axis = tuple(axis)
            r = np.amin(np.abs(x), axis=axis, keepdims=keepdims)
        elif porder == 0:
            axis = tuple(axis)
            r = x.astype(bool)
            r = np.sum(r, axis, keepdims=keepdims)
        elif porder == 1:
            axis = tuple(axis)
            r = np.sum(np.abs(x), axis, keepdims=keepdims)
        else:
            axis = tuple(axis)
            xp = np.power(np.abs(x), porder)
            s = np.sum(xp, axis=axis, keepdims=keepdims)
            r = np.power(s, 1.0 / porder)
    else:
        if isinstance(axis, list):
            axis = tuple(axis)
        r = np.linalg.norm(x, ord=porder, axis=axis, keepdims=keepdims)
    r = r.astype(x.dtype)

    return r


class XPUTestPNormOp(XPUOpTestWrapper):

    def __init__(self):
        self.op_name = 'p_norm'
        self.use_dynamic_create_class = False

    class TestXPUPNormOp(XPUOpTest):

        def setUp(self):
            self.op_type = "p_norm"
            self.dtype = self.in_type
            self.shape = [2, 3, 4, 5]
            self.epsilon = 1e-12
            self.axis = 1
            self.porder = 2.0
            self.asvector = False
            self.keepdims = False
            self.set_attrs()
            np.random.seed(12345)

            x_np = np.random.uniform(-10, 10, self.shape).astype(self.dtype)

            ref_y_np = ref_p_norm(x_np, self.axis, self.porder, self.keepdims,
                                  self.asvector)
            self.inputs = {'X': x_np}
            self.outputs = {'Out': ref_y_np}
            self.attrs = {
                'epsilon': self.epsilon,
                'axis': self.axis,
                'porder': float(self.porder),
                'asvector': self.asvector
            }

        def set_attrs(self):
            pass

        def test_check_output(self):
            self.check_output_with_place(paddle.XPUPlace(0), atol=1e-4)

        def test_check_grad(self):
            self.check_grad_with_place(paddle.XPUPlace(0), ['X'], 'Out')

    class TestPnormOp2(TestXPUPNormOp):

        def set_attrs(self):
            self.shape = [3, 20, 3]
            self.axis = 2
            self.porder = 2.0

    class TestPnormOp3(TestXPUPNormOp):

        def set_attrs(self):
            self.shape = [3, 20, 3]
            self.axis = 2
            self.porder = np.inf

    class TestPnormOp4(TestXPUPNormOp):

        def set_attrs(self):
            self.shape = [3, 20, 3]
            self.axis = 2
            self.porder = -np.inf

    class TestPnormOp5(TestXPUPNormOp):

        def set_attrs(self):
            self.shape = [3, 20, 3]
            self.axis = 2
            self.porder = 0

    class TestPnormOp6(TestXPUPNormOp):

        def set_attrs(self):
            self.shape = [3, 20, 3]
            self.axis = -1
            self.porder = 2

    class TestPnormOp7(TestXPUPNormOp):

        def set_attrs(self):
            self.shape = [3, 20, 3, 10]
            self.axis = 2
            self.porder = 2.0

    class TestPnormOp8(TestXPUPNormOp):

        def set_attrs(self):
            self.shape = [3, 20, 3]
            self.axis = 2
            self.porder = np.inf

    class TestPnormOp9(TestXPUPNormOp):

        def set_attrs(self):
            self.shape = [3, 20, 3, 10]
            self.axis = 1
            self.porder = -np.inf

    class TestPnormOp10(TestXPUPNormOp):

        def set_attrs(self):
            self.shape = [3, 20, 3, 10]
            self.axis = 2
            self.porder = 0

    class TestPnormOp11(TestXPUPNormOp):

        def set_attrs(self):
            self.shape = [3, 20, 3, 10]
            self.axis = -1
            self.porder = 2


support_types = get_xpu_op_support_types('p_norm')
for stype in support_types:
    create_test_class(globals(), XPUTestPNormOp, stype)

if __name__ == "__main__":
    unittest.main()
