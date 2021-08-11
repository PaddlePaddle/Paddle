# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.fluid.tests.unittests.op_test import OpTest, convert_float_to_uint16
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid import compiler, Program, program_guard

paddle.enable_static()
SEED = 2049
np.random.seed(SEED)


class TestMatrixRankOP(OpTest):
    def setUp(self):

        self.place = paddle.CPUPlace()
        self.op_type = "matrix_rank"
        self.init_data()

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(self.x)}
        # if (self.tol):
        self.attrs = {'tol': self.tol, 'hermitian': self.hermitian}
        # else:
        #     self.attrs = {'hermitian': self.hermitian}
        self.outputs = {'Out': self.out}

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def init_data(self):
        self.x = np.eye(3, dtype=np.float32)
        self.tol = 0.1
        self.hermitian = True
        self.out = np.linalg.matrix_rank(self.x, self.tol)


class TestMatrixRankOP1(TestMatrixRankOP):
    def init_data(self):
        self.x = np.eye(3, k=1, dtype=np.float64)
        self.tol = 0.0
        self.hermitian = False
        self.out = np.linalg.matrix_rank(self.x, self.tol)


class TestMatrixRankOP2(TestMatrixRankOP):
    def init_data(self):
        self.x = np.random.rand(3, 4, 5, 6)
        self.tol = 0.1
        self.hermitian = False
        self.out = np.linalg.matrix_rank(self.x, self.tol)


if __name__ == '__main__':
    unittest.main()
