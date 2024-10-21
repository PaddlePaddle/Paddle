#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
from scipy.special import erf

import paddle
import paddle.base.dygraph as dg
from paddle import base, static

paddle.enable_static()


class TestErfOp(OpTest):
    def setUp(self):
        self.op_type = "erf"
        self.prim_op_type = "prim"
        self.public_python_api = paddle.erf
        self.python_api = paddle.erf
        self.dtype = self._init_dtype()
        self.init_shape()
        x = np.random.uniform(-1, 1, size=self.x_shape).astype(self.dtype)
        y_ref = erf(x).astype(self.dtype)
        self.inputs = {'X': x}
        self.outputs = {'Out': y_ref}

    def init_shape(self):
        self.x_shape = [11, 17]

    def _init_dtype(self):
        return "float64"

    def test_check_output(self):
        self.check_output(check_pir=True, check_symbol_infer=False)

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', check_prim=True, check_pir=True)

    def test_check_grad_prim_pir(self):
        # Todo(CZ): float64 loss greater than 1e-8
        if self.dtype == "float64":
            self.dtype = "float32"
            self.rev_comp_atol = 1e-7
            self.rev_comp_rtol = 1e-7
        self.check_grad(['X'], 'Out', check_prim_pir=True)


class TestErfOp_ZeroDim(TestErfOp):
    def init_shape(self):
        self.x_shape = []


class TestErfLayer(unittest.TestCase):
    def setUp(self):
        self.x = np.random.uniform(-1, 1, size=(11, 17)).astype(np.float64)
        self.y = erf(self.x)

    def _test_dygraph(self, place):
        with dg.guard(place) as g:
            x_var = paddle.to_tensor(self.x)
            y_var = paddle.erf(x_var)
            y_test = y_var.numpy()
        np.testing.assert_allclose(self.y, y_test, rtol=1e-05)

    def test_dygraph(self):
        self._test_dygraph(base.CPUPlace())
        if base.is_compiled_with_cuda():
            self._test_dygraph(base.CUDAPlace(0))

    def _test_static(self, place):
        mp, sp = static.Program(), static.Program()
        with static.program_guard(mp, sp):
            x = static.data("x", shape=[11, 17], dtype="float64")
            y = paddle.erf(x)

        exe = static.Executor(place)
        exe.run(sp)
        [y_np] = exe.run(mp, feed={"x": self.x}, fetch_list=[y])
        np.testing.assert_allclose(self.y, y_np, rtol=1e-05)

    def test_static(self):
        self._test_static(base.CPUPlace())
        if base.is_compiled_with_cuda():
            self._test_static(base.CUDAPlace(0))


class TestErfFP16OP(OpTest):
    def setUp(self):
        self.op_type = "erf"
        self.prim_op_type = "prim"
        self.public_python_api = paddle.erf
        self.python_api = paddle.erf
        self.dtype = np.float16
        self.x_shape = [11, 17]
        x = np.random.uniform(-1, 1, size=self.x_shape).astype(self.dtype)
        y_ref = erf(x).astype(self.dtype)
        self.inputs = {'X': x}
        self.outputs = {'Out': y_ref}

    def test_check_output(self):
        self.check_output(check_pir=True, check_symbol_infer=False)

    def test_check_grad(self):
        self.check_grad(
            ['X'],
            'Out',
            check_prim=True,
            check_pir=True,
            check_prim_pir=True,
        )


@unittest.skipIf(
    not paddle.base.core.is_compiled_with_cuda()
    or not paddle.base.core.is_bfloat16_supported(
        paddle.base.core.CUDAPlace(0)
    ),
    "core is not compiled with CUDA and not support the bfloat16",
)
class TestErfBF16OP(OpTest):
    def setUp(self):
        self.op_type = "erf"
        self.prim_op_type = "prim"
        self.public_python_api = paddle.erf
        self.python_api = paddle.erf
        self.dtype = np.uint16
        self.x_shape = [11, 17]
        x = np.random.uniform(-1, 1, size=self.x_shape).astype(np.float32)
        y_ref = erf(x).astype(np.float32)
        self.inputs = {'X': convert_float_to_uint16(x)}
        self.outputs = {'Out': convert_float_to_uint16(y_ref)}

    def test_check_output(self):
        place = paddle.base.core.CUDAPlace(0)
        self.check_output_with_place(
            place, check_pir=True, check_symbol_infer=False
        )

    def test_check_grad(self):
        place = paddle.base.core.CUDAPlace(0)
        self.check_grad_with_place(
            place,
            ['X'],
            'Out',
            check_prim=True,
            check_pir=True,
            check_prim_pir=True,
        )


if __name__ == '__main__':
    unittest.main()
