# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest
import paddle
import paddle.fluid as fluid


def p_norm(x, axis, porder, keepdims=False):
    if axis is None: axis = -1
    xp = np.power(np.abs(x), porder)
    s = np.sum(xp, axis=axis, keepdims=keepdims)
    r = np.power(s, 1.0 / porder)
    return r


def frobenius_norm(x, axis=None, keepdims=False):
    if isinstance(axis, list): axis = tuple(axis)
    if axis is None: axis = (-2, -1)
    r = np.linalg.norm(x, ord='fro', axis=axis, keepdims=keepdims)
    return r


class TestFrobeniusNormOp(OpTest):
    def setUp(self):
        self.op_type = "frobenius_norm"
        self.init_test_case()
        x = (np.random.random(self.shape) + 1.0).astype(self.dtype)
        norm = frobenius_norm(x, self.axis, self.keepdim)
        self.reduce_all = (len(self.axis) == len(self.shape))
        self.inputs = {'X': x}
        self.attrs = {
            'dim': list(self.axis),
            'keep_dim': self.keepdim,
            'reduce_all': self.reduce_all
        }
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


class TestFrobeniusNormOp2(TestFrobeniusNormOp):
    def init_test_case(self):
        self.shape = [5, 5, 5]
        self.axis = (0, 1)
        self.keepdim = True
        self.dtype = "float32"

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestPnormOp(OpTest):
    def setUp(self):
        self.op_type = "p_norm"
        self.init_test_case()
        x = (np.random.random(self.shape) + 0.5).astype(self.dtype)
        norm = p_norm(x, self.axis, self.porder, self.keepdim)
        self.inputs = {'X': x}
        self.attrs = {
            'epsilon': self.epsilon,
            'axis': self.axis,
            'keepdim': self.keepdim,
            'porder': float(self.porder)
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
        self.porder = 2.0
        self.keepdim = False
        self.dtype = "float64"


class TestPnormOp2(TestPnormOp):
    def init_test_case(self):
        self.shape = [3, 20, 3]
        self.axis = 2
        self.epsilon = 1e-12
        self.porder = 2.0
        self.keepdim = True
        self.dtype = "float32"

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


if __name__ == '__main__':
    unittest.main()
