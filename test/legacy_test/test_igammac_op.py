# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from scipy import special

import paddle
from paddle.base import core


def ref_igammac(x, a):
    return special.gammainc(a, x)


class TestIgammaOp(OpTest):
    def setUp(self):
        self.op_type = 'igammac'
        self.python_api = paddle.igammac
        self.init_dtype_type()
        self.shape = (3, 40)
        self.x = np.random.random(self.shape).astype(self.dtype) + 1
        self.a = np.random.random(self.shape).astype(self.dtype) + 1
        self.inputs = {'x': self.x, 'a': self.a}
        out = ref_igammac(self.x, self.a)
        self.outputs = {'out': out}

    def init_dtype_type(self):
        self.dtype = np.float64

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad(self):
        self.check_grad(['x'], 'out', check_pir=True)


class TestIgammaOpFp32(TestIgammaOp):
    def init_dtype_type(self):
        self.dtype = np.float32

    def test_check_grad(self):
        self.check_grad(['x'], 'out', numeric_grad_delta=0.01, check_pir=True)


class TestIgammaOpApi(unittest.TestCase):
    def setUp(self):
        self.shape = [2, 3, 4, 5]
        self.init_dtype_type()
        self.x_np = np.random.random(self.shape).astype(self.dtype) + 1
        self.a_np = np.random.random(self.shape).astype(self.dtype) + 1
        self.place = (
            paddle.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def init_dtype_type(self):
        self.dtype = "float64"

    def test_static_api(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('x', self.x_np.shape, self.x_np.dtype)
            a = paddle.static.data('a', self.a_np.shape, self.x_np.dtype)
            out = paddle.igammac(x, a)
            exe = paddle.static.Executor(self.place)
            (res,) = exe.run(
                feed={'x': self.x_np, 'a': self.a_np}, fetch_list=[out]
            )
        out_ref = ref_igammac(self.x_np, self.a_np)
        np.testing.assert_allclose(out_ref, res, rtol=1e-5, atol=1e-5)

    def test_dygraph_api(self):
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x_np)
        a = paddle.to_tensor(self.a_np)
        out = paddle.igammac(x, a)
        out_ref = ref_igammac(self.x_np, self.a_np)
        np.testing.assert_allclose(out_ref, out.numpy(), rtol=1e-5, atol=1e-5)
        paddle.enable_static()


class TestIgammaOpApiFp32(TestIgammaOpApi):
    def init_dtype_type(self):
        self.dtype = "float32"


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
