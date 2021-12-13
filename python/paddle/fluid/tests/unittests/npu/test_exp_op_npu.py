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
from scipy.special import expit, erf

from paddle.fluid.tests.unittests.op_test import OpTest, convert_float_to_uint16
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid import compiler, Program, program_guard

paddle.enable_static()
SEED = 2049


class TestExpNPUOP(OpTest):
    def setUp(self):

        self.set_npu()
        self.place = paddle.NPUPlace(0)
        self.op_type = "exp"
        self.init_dtype()
        self.init_kernel_type()

        np.random.seed(SEED)
        x = np.random.uniform(0.1, 1, [11, 17]).astype(self.dtype)
        out = np.exp(x)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad_with_place(self.place, ['X'], 'Out')

    def init_dtype(self):
        self.dtype = np.float32

    def init_kernel_type(self):
        pass

    def set_npu(self):
        self.__class__.use_npu = True


class TestExpNPUOPFloat64(TestExpNPUOP):
    def init_dtype(self):
        self.dtype = np.float64


if __name__ == "__main__":
    unittest.main()
