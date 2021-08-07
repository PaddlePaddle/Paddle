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

import numpy as np
import unittest
import sys
sys.path.append("..")
from op_test import OpTest
import paddle
import paddle.fluid as fluid
from paddle.fluid import core

paddle.enable_static()
SEED = 2121

class TestLeakyRelu(OpTest):
    def setUp(self):
        self.set_npu()
        self.place = paddle.NPUPlace(0)
        self.op_type = "leaky_relu"
        self.init_dtype()

        x = np.random.random([20, 20]).astype(self.dtype)
        negative_slope = np.float32(0.01) 
        out = np.where(x >= 0, x, x * negative_slope).astype(self.dtype)
        
        self.inputs = {'X': x}

        self.attrs = {'negative_slope': negative_slope}
        self.outputs = {'Out': out}

    def set_npu(self):
        self.__class__.use_npu = True

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place)
    """
    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad_with_place(
            self.place, ['X'], 'Out', max_relative_error=0.01)
    """

class TestLeakyRelu16(TestLeakyRelu):
    def init_dtype(self):
        self.dtype = np.float16

class TestLeakydouble(TestLeakyRelu):
    def init_dtype(self):
        self.dtype = np.double

if __name__ == '__main__':
    unittest.main()
