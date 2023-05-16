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
from eager_op_test import OpTest, convert_float_to_uint16
from scipy.special import erfinv

import paddle
from paddle.fluid import core

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
        self.gradient = (
            np.sqrt(np.pi) / 2 * np.exp(np.square(self.res_ref)) * self.grad_out
        )
        self.inputs = {'X': self.x}
        self.outputs = {'Out': self.res_ref}

    def init_dtype(self):
        self.dtype = np.float64

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(
            ['X'],
            'Out',
            user_defined_grads=[self.gradient],
            user_defined_grad_outputs=self.grad_out,
        )


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
                x = paddle.static.data('x', [1, 5], dtype=self.dtype)
                out = paddle.erfinv(x)
                exe = paddle.static.Executor(place)
                res = exe.run(feed={'x': self.x.reshape([1, 5])})
            for r in res:
                np.testing.assert_allclose(self.res_ref, r, rtol=1e-05)

        for place in self.place:
            run(place)

    def test_dygraph_api(self):
        def run(place):
            paddle.disable_static(place)
            x = paddle.to_tensor(self.x)
            out = paddle.erfinv(x)
            np.testing.assert_allclose(self.res_ref, out.numpy(), rtol=1e-05)
            paddle.enable_static()

        for place in self.place:
            run(place)

    def test_inplace_api(self):
        def run(place):
            paddle.disable_static(place)
            x = paddle.to_tensor(self.x)
            x.erfinv_()
            np.testing.assert_allclose(self.res_ref, x.numpy(), rtol=1e-05)
            paddle.enable_static()

        for place in self.place:
            run(place)


class TestErfinvFP16OP(OpTest):
    def setUp(self):
        self.op_type = "erfinv"
        self.python_api = paddle.erfinv
        self.init_dtype()
        self.shape = [11, 17]
        self.x = np.random.uniform(-1, 1, size=self.shape).astype(self.dtype)
        self.res_ref = erfinv(self.x).astype(self.dtype)
        self.grad_out = np.ones(self.shape, self.dtype)
        self.gradient = (
            np.sqrt(np.pi) / 2 * np.exp(np.square(self.res_ref)) * self.grad_out
        )
        self.inputs = {'X': self.x}
        self.outputs = {'Out': self.res_ref}

    def init_dtype(self):
        self.dtype = np.float16

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(
            ['X'],
            'Out',
            user_defined_grads=[self.gradient],
            user_defined_grad_outputs=self.grad_out,
        )


@unittest.skipIf(
    not paddle.fluid.core.is_compiled_with_cuda()
    or not paddle.fluid.core.is_bfloat16_supported(
        paddle.fluid.core.CUDAPlace(0)
    ),
    "core is not complied with CUDA and not support the bfloat16",
)
class TestErfinvBF16OP(OpTest):
    def setUp(self):
        self.op_type = "erfinv"
        self.python_api = paddle.erfinv
        self.init_dtype()
        self.shape = [11, 17]
        self.x = np.random.uniform(-1, 1, size=self.shape).astype(np.float32)
        self.out = erfinv(self.x).astype(np.float32)
        self.res_ref = convert_float_to_uint16(self.out)
        self.grad_out = np.ones(self.shape, self.dtype)
        self.gradient = (
            np.sqrt(np.pi) / 2 * np.exp(np.square(self.res_ref)) * self.grad_out
        )
        self.inputs = {'X': convert_float_to_uint16(self.x)}
        self.outputs = {'Out': self.res_ref}

    def init_dtype(self):
        self.dtype = np.uint16

    def test_check_output(self):
        place = paddle.fluid.core.CUDAPlace(0)
        self.check_output_with_place(place)

    def test_check_grad(self):
        place = paddle.fluid.core.CUDAPlace(0)
        self.check_grad(
            place,
            ['X'],
            'Out',
            user_defined_grads=[self.gradient],
            user_defined_grad_outputs=self.grad_out,
        )


if __name__ == "__main__":
    unittest.main()
