#  Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import unittest
import sys

sys.path.append("..")
import numpy as np
from op_test_xpu import XPUOpTest

paddle.enable_static()


def ref_logsumexp(x, axis=None, keepdim=False, reduce_all=False):
    if isinstance(axis, int):
        axis = (axis,)
    elif isinstance(axis, list):
        axis = tuple(axis)
    if reduce_all:
        axis = None
    out = np.log(np.exp(x).sum(axis=axis, keepdims=keepdim))
    return out


class XPUTestLogsumexp(XPUOpTest):
    def setUp(self):
        self.op_type = 'logsumexp'
        self.shape = [2, 3, 4, 5]
        self.dtype = 'float32'
        self.axis = [-1]
        self.keepdim = False
        self.reduce_all = False
        self.set_attrs()

        np.random.seed(10)
        x = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        out = ref_logsumexp(x, self.axis, self.keepdim, self.reduce_all)

        self.inputs = {'X': x}
        self.outputs = {'Out': out}
        self.attrs = {
            'axis': self.axis,
            'keepdim': self.keepdim,
            'reduce_all': self.reduce_all,
        }

    def set_attrs(self):
        pass

    def test_check_output(self):
        if paddle.is_compiled_with_xpu():
            place = paddle.XPUPlace(0)
            self.check_output_with_place(place)

    def test_check_grad(self):
        pass


class TestLogsumexp_shape(XPUTestLogsumexp):
    def set_attrs(self):
        self.shape = [4, 5, 6]


class TestLogsumexp_axis(XPUTestLogsumexp):
    def set_attrs(self):
        self.axis = [0, -1]


class TestLogsumexp_axis_all(XPUTestLogsumexp):
    def set_attrs(self):
        self.axis = [0, 1, 2, 3]


class TestLogsumexp_keepdim(XPUTestLogsumexp):
    def set_attrs(self):
        self.keepdim = True


class TestLogsumexp_reduce_all(XPUTestLogsumexp):
    def set_attrs(self):
        self.reduce_all = True


if __name__ == '__main__':
    unittest.main()
