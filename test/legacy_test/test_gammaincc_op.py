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
from utils import static_guard

import paddle
from paddle.base import core


def ref_gammaincc(x, y):
    return special.gammaincc(x, y)


class TestGammainccOp(OpTest):
    def setUp(self):
        self.op_type = 'gammaincc'
        self.python_api = paddle.gammaincc
        self.init_dtype_type()
        self.shape = (3, 40)
        self.x = np.random.random(self.shape).astype(self.dtype) + 1
        self.y = np.random.random(self.shape).astype(self.dtype) + 1
        self.inputs = {'x': self.x, 'y': self.y}
        out = ref_gammaincc(self.x, self.y)
        self.outputs = {'out': out}

    def init_dtype_type(self):
        self.dtype = np.float64

    def test_check_output(self):
        self.check_output(check_pir=True, check_symbol_infer=False)

    def test_check_grad(self):
        self.check_grad(['y'], 'out', check_pir=True)


class TestGammainccOpFp32(TestGammainccOp):
    def init_dtype_type(self):
        self.dtype = np.float32


class TestGammainccOpApi(unittest.TestCase):
    def setUp(self):
        self.shape = [2, 3, 4, 5]
        self.init_dtype_type()
        self.x_np = np.random.random(self.shape).astype(self.dtype) + 1
        self.y_np = np.random.random(self.shape).astype(self.dtype) + 1
        self.place = (
            paddle.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def init_dtype_type(self):
        self.dtype = "float64"

    def test_static_api(self):
        with static_guard():
            with paddle.static.program_guard(paddle.static.Program()):
                x = paddle.static.data('x', self.x_np.shape, self.x_np.dtype)
                y = paddle.static.data('y', self.y_np.shape, self.y_np.dtype)
                out = paddle.gammaincc(x, y)
                exe = paddle.static.Executor(self.place)
                (res,) = exe.run(
                    feed={'x': self.x_np, 'y': self.y_np}, fetch_list=[out]
                )
            out_ref = ref_gammaincc(self.x_np, self.y_np)
            np.testing.assert_allclose(out_ref, res, rtol=1e-6, atol=1e-6)

    def test_dygraph_api(self):
        x = paddle.to_tensor(self.x_np)
        y = paddle.to_tensor(self.y_np)
        out = paddle.gammaincc(x, y)
        out_ref = ref_gammaincc(self.x_np, self.y_np)
        np.testing.assert_allclose(out_ref, out.numpy(), rtol=1e-6, atol=1e-6)

    def test_x_le_zero_error(self):
        x = paddle.to_tensor(self.x_np)
        y = paddle.to_tensor(self.y_np)
        x[0] = -1
        self.assertRaises(ValueError, paddle.gammaincc, x, y)

    def test_a_le_zero_error(self):
        x = paddle.to_tensor(self.x_np)
        y = paddle.to_tensor(self.y_np)
        y[0] = -1
        self.assertRaises(ValueError, paddle.gammaincc, x, y)

    def test_dtype_error(self):
        with static_guard():
            # in static graph mode
            with self.assertRaises(TypeError):
                with paddle.static.program_guard(paddle.static.Program()):
                    x = paddle.static.data(
                        name="x", shape=self.shape, dtype="int32"
                    )
                    y = paddle.static.data(
                        name="y", shape=self.shape, dtype="int32"
                    )
                    out = paddle.gammaincc(x, y)

        # in dynamic mode
        with self.assertRaises(RuntimeError):
            with paddle.base.dygraph.guard():
                x = paddle.to_tensor(self.x_np, dtype="int32")
                y = paddle.to_tensor(self.y_np, dtype="int32")
                res = paddle.gammaincc(x, y)


class TestGammainccOpFp32Api(TestGammainccOpApi):
    def init_dtype_type(self):
        self.dtype = "float32"


if __name__ == "__main__":
    unittest.main()
