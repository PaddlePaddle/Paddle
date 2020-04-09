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

from __future__ import print_function

import unittest
import numpy as np
from op_test import OpTest, skip_check_grad_ci


def p_norm(x, axis, porder, epsilon, keepdims=False):
    xp = np.power(np.abs(x), porder)
    s = np.sum(xp, axis=axis, keepdims=keepdims)
    r = np.power(s + epsilon, 1.0 / porder)
    return r


class TestPnormOp(OpTest):
    def setUp(self):
        self.op_type = "p_norm"
        self.init_test_case()
        x = (np.random.random(self.shape) + 0.5).astype(self.dtype)
        norm = p_norm(x, self.axis, self.porder, self.epsilon, self.keepdim)
        self.inputs = {'X': x}
        self.attrs = {
            'epsilon': self.epsilon,
            'axis': self.axis,
            'keepdim': self.keepdim,
            'porder': self.porder
        }
        self.outputs = {'Out': norm}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')

    def init_test_case(self):
        self.shape = [2, 3, 4, 5]
        self.axis = 1
        self.epsilon = 1e-12
        self.porder = 2
        self.keepdim = False
        self.dtype = "float64"


@skip_check_grad_ci(reason="'check_grad' on large inputs is too slow, " +
                    "however it is desirable to cover the forward pass")
class TestPnormOp2(TestPnormOp):
    def init_test_case(self):
        self.shape = [128, 1024, 14, 14]
        self.axis = 2
        self.epsilon = 1e-12
        self.porder = 2
        self.keepdim = True
        self.dtype = "float32"

    def test_check_grad(self):
        pass


if __name__ == '__main__':
    unittest.main()
