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
from op_test import OpTest, convert_float_to_uint16

import paddle
from paddle.base import core
from paddle.pir_utils import test_with_pir_api

paddle.enable_static()
np.random.seed(0)


def atan2_grad(x1, x2, dout):
    dx1 = dout * x2 / (x1 * x1 + x2 * x2)
    dx2 = -dout * x1 / (x1 * x1 + x2 * x2)
    return dx1, dx2


class TestAtan2(OpTest):
    def setUp(self):
        self.op_type = "atan2"
        self.python_api = paddle.atan2
        self.check_cinn = True
        self.init_dtype()

        x1 = np.random.uniform(-1, -0.1, [15, 17]).astype(self.dtype)
        x2 = np.random.uniform(0.1, 1, [15, 17]).astype(self.dtype)
        out = np.arctan2(x1, x2)

        self.inputs = {'X1': x1, 'X2': x2}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        self.check_grad(
            ['X1', 'X2'], 'Out', check_cinn=self.check_cinn, check_pir=True
        )

    def test_check_output(self):
        self.check_output(check_cinn=self.check_cinn, check_pir=True)

    def init_dtype(self):
        self.dtype = np.float64


class TestAtan2_float(TestAtan2):
    def init_dtype(self):
        self.dtype = np.float32

    def test_check_grad(self):
        if self.dtype not in [np.int32, np.int64]:
            self.check_grad(
                ['X1', 'X2'],
                'Out',
                user_defined_grads=atan2_grad(
                    self.inputs['X1'],
                    self.inputs['X2'],
                    1 / self.inputs['X1'].size,
                ),
                check_cinn=self.check_cinn,
                check_pir=True,
            )


class TestAtan2_float16(TestAtan2_float):
    def init_dtype(self):
        self.dtype = np.float16


class TestAtan2_int32(TestAtan2_float):
    def init_dtype(self):
        self.dtype = np.int32


class TestAtan2_int64(TestAtan2_float):
    def init_dtype(self):
        self.dtype = np.int64


class TestAtan2API(unittest.TestCase):
    def init_dtype(self):
        self.dtype = 'float64'
        self.shape = [11, 17]

    def setUp(self):
        self.init_dtype()
        self.x1 = np.random.uniform(0.1, 1, self.shape).astype(self.dtype)
        self.x2 = np.random.uniform(-1, -0.1, self.shape).astype(self.dtype)
        self.place = [paddle.CPUPlace()]
        if core.is_compiled_with_cuda():
            self.place.append(paddle.CUDAPlace(0))

    @test_with_pir_api
    def test_static_api(self):
        paddle.enable_static()

        def run(place):
            with paddle.static.program_guard(paddle.static.Program()):
                X1 = paddle.static.data('X1', self.shape, dtype=self.dtype)
                X2 = paddle.static.data('X2', self.shape, dtype=self.dtype)
                out = paddle.atan2(X1, X2)
                exe = paddle.static.Executor(place)
                res = exe.run(feed={'X1': self.x1, 'X2': self.x2})
            out_ref = np.arctan2(self.x1, self.x2)
            for r in res:
                np.testing.assert_allclose(out_ref, r, rtol=1e-05)

        for place in self.place:
            run(place)

    def test_dygraph_api(self):
        def run(place):
            paddle.disable_static(place)
            X1 = paddle.to_tensor(self.x1)
            X2 = paddle.to_tensor(self.x2)
            out = paddle.atan2(X1, X2)
            out_ref = np.arctan2(self.x1, self.x2)
            np.testing.assert_allclose(out_ref, out.numpy(), rtol=1e-05)
            paddle.enable_static()

        for place in self.place:
            run(place)


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and not support the bfloat16",
)
class TestAtan2BF16OP(OpTest):
    def setUp(self):
        self.op_type = 'atan2'
        self.python_api = paddle.atan2
        self.dtype = np.uint16
        self.check_cinn = True
        x1 = np.random.uniform(-1, -0.1, [15, 17]).astype('float32')
        x2 = np.random.uniform(0.1, 1, [15, 17]).astype('float32')
        out = np.arctan2(x1, x2)

        self.inputs = {
            'X1': convert_float_to_uint16(x1),
            'X2': convert_float_to_uint16(x2),
        }
        self.outputs = {'Out': convert_float_to_uint16(out)}

    def test_check_output(self):
        place = core.CUDAPlace(0)
        self.check_output_with_place(
            place, check_cinn=self.check_cinn, check_pir=True
        )

    def test_check_grad(self):
        place = core.CUDAPlace(0)
        self.check_grad_with_place(
            place,
            ['X1', 'X2'],
            'Out',
            check_cinn=self.check_cinn,
            check_pir=True,
        )


class TestAtan2Error(unittest.TestCase):
    @test_with_pir_api
    def test_mismatch(self):
        paddle.enable_static()

        def test_mismatch_numel():
            X = paddle.static.data('X', (1,), dtype=np.float64)
            Y = paddle.static.data('Y', (0,), dtype=np.float64)
            out = paddle.atan2(X, Y)

        self.assertRaises(ValueError, test_mismatch_numel)


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
