#  Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import sys
sys.path.append("..")
import unittest
import numpy as np
import paddle
import paddle.fluid as fluid
from op_test import OpTest, skip_check_grad_ci

paddle.enable_static()
SEED = 2021


def l2_norm(x, axis, epsilon):
    x2 = x**2
    s = np.sum(x2, axis=axis, keepdims=True)
    r = np.sqrt(s) + epsilon
    y = x / np.broadcast_to(r, x.shape)
    return y, r


class TestNorm(OpTest):
    def setUp(self):
        paddle.enable_static()
        self.set_npu()
        self.place = paddle.NPUPlace(0)
        self.op_type = "norm"
        self.init_dtype()

        x = np.random.random(self.shape).astype(self.dtype)
        y, norm = l2_norm(x, self.axis, self.epsilon)
        self.inputs = {'X': x}
        self.attrs = {'epsilon': self.epsilon, 'axis': self.axis}
        self.outputs = {'Out': y, 'Norm': norm}

    def set_npu(self):
        self.__class__.use_npu = True

    def init_dtype(self):
        self.dtype = np.float32
        self.axis = -1
        self.epsilon = 1e-10
        self.shape = (2, 3, 4, 100)

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ['X'], 'Out')


""" Note(@xiongkun03) : This Test case may not always pass. 
                        May Raise :
                        Expected [ 0.0135 ] but got [ 0.014 ]
                        This is caused by Float16 round-off error in forward procedure. 
                        May effect the accuracy of some model. recommend usage of float32 instead of float16.
"""


class TestNormFP16(OpTest):
    def setUp(self):
        paddle.enable_static()
        self.set_npu()
        self.place = paddle.NPUPlace(0)
        self.op_type = "norm"
        self.init_dtype()

        x = np.random.random(self.shape).astype(self.dtype)
        y, norm = l2_norm(x, self.axis, self.epsilon)
        y, norm = y.astype(self.dtype), norm.astype(self.dtype)
        self.inputs = {'X': x}
        self.attrs = {'epsilon': self.epsilon, 'axis': self.axis}
        self.outputs = {'Out': y, 'Norm': norm}

    def set_npu(self):
        self.__class__.use_npu = True
        self.__class__.no_need_check_grad = True

    def init_dtype(self):
        self.dtype = np.float16
        self.axis = -1
        self.epsilon = 1e-10
        self.shape = (2, 3, 100)

    def test_check_output(self):
        self.check_output_with_place(self.place)


if __name__ == '__main__':
    unittest.main()
