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
import paddle
import paddle.fluid.core as core
import sys
sys.path.append("..")
from op_test import OpTest
from paddle.fluid import Program, program_guard
import paddle.fluid.dygraph as dg
from numpy.random import random as rand

paddle.enable_static()


class TestConjOp(OpTest):
    def setUp(self):
        self.op_type = "conj"
        self.init_dtype_type()

        x = (np.random.random((12, 14)) + 1j * np.random.random(
            (12, 14))).astype(self.dtype)
        out = np.conj(x)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def init_dtype_type(self):
        self.dtype = np.complex64

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['X'], 'Out')


class TestComplexConjOp(unittest.TestCase):
    def setUp(self):
        self._dtypes = ["float32", "float64"]
        self._places = [paddle.CPUPlace()]
        if paddle.is_compiled_with_cuda():
            self._places.append(paddle.CUDAPlace(0))

    def test_conj_api(self):
        for dtype in self._dtypes:
            input = rand([2, 20, 2, 3]).astype(dtype) + 1j * rand(
                [2, 20, 2, 3]).astype(dtype)
            for place in self._places:
                with dg.guard(place):
                    var_x = dg.to_variable(input)
                    result = paddle.conj(var_x).numpy()
                    target = np.conj(input)
                    self.assertTrue(np.allclose(result, target))


if __name__ == "__main__":
    unittest.main()
