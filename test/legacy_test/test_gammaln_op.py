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
from op_test import OpTest, convert_float_to_uint16
from scipy import special

import paddle
from paddle.base import core


def ref_gammaln(x):
    return special.gammaln(x)


def ref_gammaln_grad(x, dout):
    return dout * special.polygamma(0, x)


class TestGammalnOp(OpTest):
    def setUp(self):
        self.op_type = 'gammaln'
        self.python_api = paddle.gammaln
        self.init_dtype_type()
        self.shape = (3, 40)
        self.x = np.random.random(self.shape).astype(self.dtype) + 1
        self.inputs = {'x': self.x}
        out = ref_gammaln(self.x)
        self.outputs = {'out': out}

    def init_dtype_type(self):
        self.dtype = np.float64

    def test_check_output(self):
        self.check_output(check_pir=True, check_symbol_infer=False)

    def test_check_grad(self):
        self.check_grad(['x'], 'out', check_pir=True)


class TestGammalnOpFp32(TestGammalnOp):
    def init_dtype_type(self):
        self.dtype = np.float32


class TestGammalnFP16Op(TestGammalnOp):
    def init_dtype_type(self):
        self.dtype = np.float16


class TestGammalnBigNumberOp(TestGammalnOp):
    def setUp(self):
        self.op_type = 'gammaln'
        self.python_api = paddle.gammaln
        self.init_dtype_type()
        self.shape = (100, 1)
        self.x = np.random.random(self.shape).astype(self.dtype) + 1
        self.x[:5, 0] = np.array([1e5, 1e10, 1e20, 1e40, 1e80])
        self.inputs = {'x': self.x}
        out = ref_gammaln(self.x)
        self.outputs = {'out': out}

    def init_dtype_type(self):
        self.dtype = np.float64

    def test_check_grad(self):
        d_out = self.outputs['out']
        d_x = ref_gammaln_grad(self.x, d_out)
        self.check_grad(
            ['x'],
            'out',
            user_defined_grads=[
                d_x,
            ],
            user_defined_grad_outputs=[
                d_out,
            ],
            check_pir=True,
        )


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA or not support bfloat16",
)
class TestGammalnBF16Op(OpTest):
    def setUp(self):
        self.op_type = 'gammaln'
        self.python_api = paddle.gammaln
        self.dtype = np.uint16
        self.shape = (5, 30)
        x = np.random.random(self.shape).astype("float32") + 1
        self.inputs = {'x': convert_float_to_uint16(x)}
        out = ref_gammaln(x)
        self.outputs = {'out': convert_float_to_uint16(out)}

    def test_check_output(self):
        self.check_output_with_place(
            core.CUDAPlace(0), check_pir=True, check_symbol_infer=False
        )

    def test_check_grad(self):
        self.check_grad_with_place(
            core.CUDAPlace(0), ['x'], 'out', check_pir=True
        )


class TestGammalnOpApi(unittest.TestCase):
    def setUp(self):
        self.shape = [2, 3, 4, 5]
        self.init_dtype_type()
        self.x_np = np.random.random(self.shape).astype(self.dtype) + 1
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
            out = paddle.gammaln(x)
            exe = paddle.static.Executor(self.place)
            (res,) = exe.run(feed={'x': self.x_np}, fetch_list=[out])
        out_ref = ref_gammaln(self.x_np)
        np.testing.assert_allclose(out_ref, res, rtol=1e-5, atol=1e-5)

    def test_dygraph_api(self):
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x_np)
        out = paddle.gammaln(x)
        out_ref = ref_gammaln(self.x_np)
        np.testing.assert_allclose(out_ref, out.numpy(), rtol=1e-5, atol=1e-5)
        paddle.enable_static()


class TestGammalnOpApiFp32(TestGammalnOpApi):
    def init_dtype_type(self):
        self.dtype = "float32"


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
