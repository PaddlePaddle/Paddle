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
from scipy.special import erfinv
from op_test import OpTest
import paddle
import paddle.fluid.core as core

paddle.enable_static()
np.random.seed(0)


class TestErfinv(OpTest):
    def setUp(self):
        self.op_type = "erfinv"
        self.python_api = paddle.erfinv
        self.init_dtype()
        self.shape = [11, 17]
        self.x = np.random.uniform(-1, 1, size=self.shape).astype(self.dtype)
        self.res_ref = erfinv(self.x).astype(self.dtype)
        self.grad_out = np.ones(self.shape, self.dtype)
        self.gradient = np.sqrt(np.pi) / 2 * np.exp(np.square(
            self.res_ref)) * self.grad_out
        self.inputs = {'X': self.x}
        self.outputs = {'Out': self.res_ref}

    def init_dtype(self):
        self.dtype = np.float64

    def test_check_output(self):
        self.check_output(check_eager=True)

    def test_check_grad(self):
        self.check_grad(
            ['X'],
            'Out',
            user_defined_grads=[self.gradient],
            user_defined_grad_outputs=self.grad_out)


class TestErfinvFP32(TestErfinv):
    def init_dtype(self):
        self.dtype = np.float32


class TestErfinvAPI(unittest.TestCase):
    def init_dtype(self):
        self.dtype = 'float32'

    def setUp(self):
        self.init_dtype()
        self.x = np.random.rand(5).astype(self.dtype)
        self.res_ref = erfinv(self.x)
        self.place = [paddle.CPUPlace()]
        if core.is_compiled_with_cuda():
            self.place.append(paddle.CUDAPlace(0))

    def test_static_api(self):
        paddle.enable_static()

        def run(place):
            with paddle.static.program_guard(paddle.static.Program()):
                x = paddle.fluid.data('x', [1, 5], dtype=self.dtype)
                out = paddle.erfinv(x)
                exe = paddle.static.Executor(place)
                res = exe.run(feed={'x': self.x.reshape([1, 5])})
            for r in res:
                self.assertEqual(np.allclose(self.res_ref, r), True)

        for place in self.place:
            run(place)

    def test_dygraph_api(self):
        def run(place):
            paddle.disable_static(place)
            x = paddle.to_tensor(self.x)
            out = paddle.erfinv(x)
            self.assertEqual(np.allclose(self.res_ref, out.numpy()), True)
            paddle.enable_static()

        for place in self.place:
            run(place)

    def test_inplace_api(self):
        def run(place):
            paddle.disable_static(place)
            x = paddle.to_tensor(self.x)
            x.erfinv_()
            self.assertEqual(np.allclose(self.res_ref, x.numpy()), True)
            paddle.enable_static()

        for place in self.place:
            run(place)


if __name__ == "__main__":
    unittest.main()
