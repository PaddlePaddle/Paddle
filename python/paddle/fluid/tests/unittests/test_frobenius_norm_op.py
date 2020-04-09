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


def frobenius_norm(x, axis, keepdims=False):
    r = np.linalg.norm(x, ord='fro', axis=axis, keepdims=keepdims)
    return r


class TestFrobeniusNormOp(OpTest):
    def setUp(self):
        self.op_type = "frobenius_norm"
        self.init_test_case()
        x = np.random.random(self.shape).astype(self.dtype)
        norm = frobenius_norm(x, self.axis, self.keepdim)
        self.inputs = {'X': x}
        self.attrs = {'dim': list(self.axis), 'keep_dim': self.keepdim}
        self.outputs = {'Out': norm}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')

    def init_test_case(self):
        self.shape = [2, 3, 4, 5]
        self.axis = (1, 2)
        self.keepdim = False
        self.dtype = "float64"


@skip_check_grad_ci(reason="'check_grad' on large inputs is too slow, " +
                    "however it is desirable to cover the forward pass")
class TestFrobeniusNormOp2(TestFrobeniusNormOp):
    def init_test_case(self):
        self.shape = [128, 128, 14, 14]
        self.axis = (0, 1)
        self.keepdim = True
        self.dtype = "float32"

    def test_check_grad(self):
        pass


if __name__ == '__main__':
    unittest.main()
