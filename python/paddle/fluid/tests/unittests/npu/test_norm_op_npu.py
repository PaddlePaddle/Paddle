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
        self.axis = 1
        self.epsilon = 1e-10
        self.shape = (2, 3, 4, 5)

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestNormOp2(TestNorm):
    def init_test_case(self):
        self.shape = [5, 3, 9, 7]
        self.axis = 0
        self.epsilon = 1e-8
        self.dtype = np.float32


class TestNormOp3(TestNorm):
    def init_test_case(self):
        self.shape = [5, 3, 2, 7]
        self.axis = -1
        self.epsilon = 1e-8
        self.dtype = np.float32


class TestNormOp4(TestNorm):
    def init_test_case(self):
        self.shape = [128, 1024, 14, 14]
        self.axis = 2
        self.epsilon = 1e-8
        self.dtype = np.float32


class API_NormTest(unittest.TestCase):
    def test_errors(self):
        paddle.enable_static()
        with fluid.program_guard(fluid.Program()):

            def test_norm_x_type():
                data = fluid.data(name="x", shape=[3, 3], dtype="float64")
                out = fluid.layers.l2_normalize(data)

            self.assertRaises(TypeError, test_norm_x_type)


class TestNormFP16(TestNorm):
    def set_npu(self):
        self.__class__.use_npu = True
        self.__class__.no_need_check_grad = True

    def init_dtype(self):
        self.dtype = np.float16
        self.axis = -1
        self.epsilon = 1e-10
        self.shape = (2, 3, 100)


if __name__ == '__main__':
    unittest.main()
