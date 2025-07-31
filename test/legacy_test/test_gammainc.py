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
from op_test import OpTest, get_device_place
from scipy import special

import paddle


def ref_gammainc(x, y):
    return special.gammainc(x, y)


class TestGammaincApi(unittest.TestCase):
    def setUp(self):
        self.shape = [2, 3, 4, 5]
        self.init_dtype_type()
        self.x_np = np.random.random(self.shape).astype(self.dtype) + 1
        self.y_np = np.random.random(self.shape).astype(self.dtype) + 1
        self.place = get_device_place()

    def init_dtype_type(self):
        self.dtype = "float64"

    def test_static_api(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('x', self.x_np.shape, self.x_np.dtype)
            y = paddle.static.data('y', self.y_np.shape, self.y_np.dtype)
            out = paddle.gammainc(x, y)
            exe = paddle.static.Executor(self.place)
            (res,) = exe.run(
                feed={'x': self.x_np, 'y': self.y_np}, fetch_list=[out]
            )
        out_ref = ref_gammainc(self.x_np, self.y_np)
        np.testing.assert_allclose(out_ref, res, rtol=1e-6, atol=1e-6)
        self.assertEqual(out.dtype, x.dtype)

    def test_dygraph_api(self):
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x_np)
        y = paddle.to_tensor(self.y_np)
        out = paddle.gammainc(x, y)
        out_ref = ref_gammainc(self.x_np, self.y_np)
        np.testing.assert_allclose(out_ref, out.numpy(), rtol=1e-6, atol=1e-6)
        self.assertEqual(out.dtype, x.dtype)
        paddle.enable_static()


class TestGammaincApiFp32(TestGammaincApi):
    def init_dtype_type(self):
        self.dtype = "float32"


class TestGammaincOp_ZeroSize(OpTest):
    def setUp(self):
        self.op_type = 'gammaincc'
        self.python_api = paddle.gammainc
        self.init_dtype_type()
        self.init_shape()
        self.x = np.random.random(self.shape).astype(self.dtype) + 1
        self.y = np.random.random(self.shape).astype(self.dtype) + 1
        self.inputs = {'x': self.x, 'y': self.y}
        out = ref_gammainc(self.x, self.y)
        self.outputs = {'out': out}

    def init_shape(self):
        self.shape = (0, 40)

    def init_dtype_type(self):
        self.dtype = np.float64

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['y'], 'out')


class TestGammaincOp_ZeroSize2(TestGammaincOp_ZeroSize):

    def init_shape(self):
        self.shape = (0,)


if __name__ == "__main__":
    unittest.main()
