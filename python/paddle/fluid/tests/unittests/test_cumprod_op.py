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
import numpy as np

from op_test import OpTest
import random
import paddle

import paddle.nn as nn
import paddle.nn.functional as F
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid import compiler, Program, program_guard

np.random.seed(0)


# test function.
class TestCumprod(OpTest):
    def setUp(self):
        self.op_type = "cumprod"
        self.init_dtype()

        x = (np.random.rand(2, 3, 10, 10) + 0.5).astype(self.dtype)
        out = np.cumprod(x, axis=1)
        self.inputs = {'X': x}
        self.outputs = {'Out': out}
        self.attrs = {'dim': 1}

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')

    def test_check_output(self):
        self.check_output()

    def init_dtype(self):
        self.dtype = np.float64


# test api.
class TestCumprodAPI(unittest.TestCase):
    def init_dtype(self):
        self.dtype = 'complex'
        self.shape = [2, 3, 10, 10]

    def setUp(self):
        self.init_dtype()
        self.x = (np.random.rand(2, 3, 10, 10) + 0.5).astype(self.dtype)
        self.place = [paddle.CPUPlace()]
        if core.is_compiled_with_cuda():
            self.place.append(paddle.CUDAPlace(0))

    # test static graph api.
    def test_static_api(self):
        paddle.enable_static()

        def run(place):
            with paddle.static.program_guard(paddle.static.Program()):
                X = paddle.fluid.data('X', self.shape, dtype=self.dtype)
                out = paddle.cumprod(X, -2)
                exe = paddle.static.Executor(place)
                res = exe.run(feed={'X': self.x})
            out_ref = np.cumprod(self.x, -2)

            for r in res:
                self.assertEqual(np.allclose(out_ref, r), True)

        for place in self.place:
            run(place)

    # test dynamic graph api.
    def test_dygraph_api(self):
        def run(place):
            paddle.disable_static(place)
            X = paddle.to_tensor(self.x)
            out = paddle.cumprod(X, 1)
            out_ref = np.cumprod(self.x, 1)
            self.assertEqual(np.allclose(out_ref, out.numpy()), True)
            paddle.enable_static()

        for place in self.place:
            run(place)


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
