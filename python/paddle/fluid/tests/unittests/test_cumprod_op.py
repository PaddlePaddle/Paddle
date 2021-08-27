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
    def prepare_inputs_outputs_attrs(self, dim, zero_num):
        x = np.random.random(self.shape).astype('float64') + 0.5
        if zero_num > 0:
            zero_num = min(zero_num, x.size)
            shape = x.shape
            x = x.flatten()
            indices = random.sample(range(x.size), zero_num)
            for i in indices:
                x[i] = 0
            x = np.reshape(x, self.shape)
        self.inputs = {'X': x}
        self.outputs = {'Out': np.cumprod(x, axis=dim)}
        self.attrs = {'dim': dim}

    def init_params(self):
        self.shape = [4, 5, 6]
        self.zero_nums = [0, 10, 20, 30, int(np.prod(self.shape))]

    def setUp(self):
        self.init_params()
        self.op_type = "cumprod"
        self.inputs = {'X': None}
        self.outputs = {'Out': None}
        self.attrs = {'dim': None}

    def _get_places(self):
        return [paddle.CUDAPlace(0)]

    # test forward.
    def test_check_output(self):
        for dim in range(-len(self.shape), len(self.shape)):
            for zero_num in self.zero_nums:
                self.prepare_inputs_outputs_attrs(dim, zero_num)
                self.check_output()

    # test backward.
    def test_check_grad(self):
        for dim in range(-len(self.shape), len(self.shape)):
            for zero_num in self.zero_nums:
                self.prepare_inputs_outputs_attrs(dim, zero_num)
                self.check_grad(['X'], 'Out')


# test api.
class TestCumprodAPI(unittest.TestCase):
    def init_dtype(self):
        self.dtype = 'float64'
        self.shape = [2, 3, 10, 10]

    def setUp(self):
        self.init_dtype()
        self.x = np.random.rand(2, 3, 10, 10) + 0.5
        self.place = [paddle.CPUPlace()]
        if core.is_compiled_with_cuda():
            self.place.append(paddle.CUDAPlace(0))

    # test static graph api.
    def test_static_api(self):
        paddle.enable_static()

        def run(place):
            with paddle.static.program_guard(paddle.static.Program()):
                X = paddle.fluid.data('X', self.shape, dtype=self.dtype)
                out = paddle.cumprod(X, -5)
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
            out = paddle.cumprod(X, 4)
            out_ref = np.cumprod(self.x, 1)
            self.assertEqual(np.allclose(out_ref, out.numpy()), True)
            paddle.enable_static()

        for place in self.place:
            run(place)


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
