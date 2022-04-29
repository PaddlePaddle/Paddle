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

import unittest
import math
import numpy as np
import paddle
from op_test import OpTest

paddle.enable_static()


class TestLgammaOp(OpTest):
    def setUp(self):
        self.op_type = 'lgamma'
        self.python_api = paddle.lgamma
        self.init_dtype_type()
        shape = (5, 20)
        data = np.random.random(shape).astype(self.dtype) + 1
        self.inputs = {'X': data}
        result = np.ones(shape).astype(self.dtype)
        for i in range(shape[0]):
            for j in range(shape[1]):
                result[i][j] = math.lgamma(data[i][j])
        self.outputs = {'Out': result}

    def init_dtype_type(self):
        self.dtype = np.float64

    def test_check_output(self):
        self.check_output(check_eager=True)

    def test_check_grad_normal(self):
        self.check_grad(['X'], 'Out', numeric_grad_delta=1e-7, check_eager=True)


class TestLgammaOpFp32(TestLgammaOp):
    def init_dtype_type(self):
        self.dtype = np.float32

    def test_check_grad_normal(self):
        self.check_grad(
            ['X'], 'Out', numeric_grad_delta=0.005, check_eager=True)


if __name__ == "__main__":
    unittest.main()
