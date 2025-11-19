# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import random
import unittest

import numpy as np
from op_test import convert_float_to_uint16, get_places, is_custom_device

import paddle
from paddle.device import get_device


def cumprod_wrapper(x, dim=-1, exclusive=False, reverse=False):
    return paddle._C_ops.cumprod(x, dim, exclusive, reverse)


# define cumprod grad function.
def cumprod_grad(x, y, dy, dx, shape, dim, exclusive=False, reverse=False):
    if dim < 0:
        dim += len(shape)
    mid_dim = shape[dim]
    outer_dim = 1
    inner_dim = 1
    for i in range(0, dim):
        outer_dim *= shape[i]
    for i in range(dim + 1, len(shape)):
        inner_dim *= shape[i]
    if not reverse:
        for i in range(outer_dim):
            for k in range(inner_dim):
                for j in range(mid_dim):
                    index = i * mid_dim * inner_dim + j * inner_dim + k
                    for n in range(mid_dim):
                        pos = i * mid_dim * inner_dim + n * inner_dim + k
                        elem = 0
                        if exclusive:
                            if pos > index:
                                elem = dy[pos] * y[index]
                                for m in range(
                                    index + inner_dim, pos, inner_dim
                                ):
                                    elem *= x[m]
                            else:
                                elem = 0
                        else:
                            if j == 0:
                                elem = dy[pos]
                            else:
                                elem = dy[pos] * y[index - inner_dim]
                            if pos > index:
                                for m in range(
                                    index + inner_dim,
                                    pos + inner_dim,
                                    inner_dim,
                                ):
                                    elem *= x[m]
                            elif pos < index:
                                elem = 0
                        dx[index] += elem
    else:
        for i in range(outer_dim):
            for k in range(inner_dim):
                for j in range(mid_dim - 1, -1, -1):
                    index = i * mid_dim * inner_dim + j * inner_dim + k
                    for n in range(mid_dim - 1, -1, -1):
                        pos = i * mid_dim * inner_dim + n * inner_dim + k
                        elem = 0
                        if exclusive:
                            if pos < index:
                                elem = dy[pos] * y[index]
                                for m in range(
                                    index - inner_dim, pos, -inner_dim
                                ):
                                    elem *= x[m]
                        else:
                            if j == mid_dim - 1:
                                elem = dy[pos]
                            else:
                                elem = dy[pos] * y[index + inner_dim]
                            if pos < index:
                                for m in range(
                                    index - inner_dim,
                                    pos - inner_dim,
                                    -inner_dim,
                                ):
                                    elem *= x[m]
                            elif pos > index:
                                elem = 0
                        dx[index] += elem


def skip_if_not_cpu_or_gpu(func):
    def wrapper(self):
        device = get_device()
        if not (
            device == 'cpu' or device.startswith('gpu:') or is_custom_device()
        ):
            self.skipTest(f"Test skipped on device: {device}")
        return func(self)

    return wrapper


class TestCumprod(unittest.TestCase):
    def init_params(self):
        self.shape = (2, 3, 4, 5)
        self.zero_nums = [0, 10, 20, 30, int(np.prod(self.shape))]

    def init_dtype(self):
        self.dtype = np.float64
        self.val_dtype = np.float64

    def setUp(self):
        paddle.disable_static()
        self.init_params()
        self.init_dtype()

    def tearDown(self):
        paddle.enable_static()

    def prepare_test_data(self, dim, zero_num):
        self.x = (
            np.random.uniform(0.0, 0.5, self.shape).astype(self.val_dtype) + 0.5
        )
        if zero_num > 0:
            zero_num = min(zero_num, self.x.size)
            shape = self.x.shape
            self.x = self.x.flatten()
            indices = random.sample(range(self.x.size), zero_num)
            for i in indices:
                self.x[i] = 0
            self.x = np.reshape(self.x, self.shape)
        self.expected_out = np.cumprod(self.x, axis=dim)

    def compute_expected_grad(self, dim):
        reshape_x = self.x.reshape(self.x.size)
        grad_out = np.ones(self.x.size, self.val_dtype)
        grad_x = np.zeros(self.x.size, self.val_dtype)
        out_data = self.expected_out.reshape(self.x.size)

        if self.dtype == np.complex128 or self.dtype == np.complex64:
            reshape_x = np.conj(reshape_x)
            out_data = np.conj(out_data)

        cumprod_grad(reshape_x, out_data, grad_out, grad_x, self.shape, dim)

        return grad_x.reshape(self.shape)

    def test_forward_computation(self):
        for dim in range(-len(self.shape), len(self.shape)):
            for zero_num in self.zero_nums:
                with self.subTest(dim=dim, zero_num=zero_num):
                    self._test_forward_for_case(dim, zero_num)

    def _test_forward_for_case(self, dim, zero_num):
        self.prepare_test_data(dim, zero_num)

        x_tensor = paddle.to_tensor(self.x, dtype=self.val_dtype)
        out = paddle.cumprod(x_tensor, dim=dim)

        np.testing.assert_allclose(
            out.numpy(), self.expected_out, rtol=1e-05, atol=1e-06
        )

    def test_gradient_computation(self):
        for dim in range(-len(self.shape), len(self.shape)):
            for zero_num in [0, 10]:
                with self.subTest(dim=dim, zero_num=zero_num):
                    self._test_gradient_for_case(dim, zero_num)

    def _test_gradient_for_case(self, dim, zero_num):
        self.prepare_test_data(dim, zero_num)

        x_tensor = paddle.to_tensor(
            self.x, dtype=self.val_dtype, stop_gradient=False
        )
        out = paddle.cumprod(x_tensor, dim=dim)

        np.testing.assert_allclose(
            out.numpy(), self.expected_out, rtol=1e-05, atol=1e-06
        )

        loss = paddle.sum(out)
        loss.backward()

        expected_grad = self.compute_expected_grad(dim)

        if self.dtype == np.float64:
            np.testing.assert_allclose(
                x_tensor.grad.numpy(), expected_grad, rtol=1e-05, atol=1e-06
            )
        else:
            if self.dtype == np.uint16:
                expected_grad_converted = convert_float_to_uint16(expected_grad)
                np.testing.assert_allclose(
                    x_tensor.grad.numpy(),
                    expected_grad_converted,
                    rtol=1e-03,
                    atol=1e-04,
                )
            else:
                np.testing.assert_allclose(
                    x_tensor.grad.numpy(), expected_grad, rtol=1e-04, atol=1e-05
                )


class TestCumprodDtypeFloat32(TestCumprod):
    def init_dtype(self):
        self.dtype = np.float32
        self.val_dtype = np.float32

    @skip_if_not_cpu_or_gpu
    def test_dtype_float32(self):
        self.prepare_test_data(dim=1, zero_num=0)

        x = paddle.to_tensor(self.x, dtype='float32')
        x.stop_gradient = False
        out = paddle.cumprod(x, dim=1, dtype='float32')
        self.assertEqual(out.dtype, paddle.float32)

        out_ref = np.cumprod(self.x.astype(np.float32), axis=1).astype(
            np.float32
        )
        np.testing.assert_allclose(out_ref, out.numpy(), rtol=1e-05)

        loss = paddle.sum(out)
        loss.backward()
        self.assertEqual(x.grad.dtype, paddle.float32)

        expected_grad = self.compute_expected_grad(1)
        np.testing.assert_allclose(
            x.grad.numpy(), expected_grad, rtol=1e-04, atol=1e-05
        )


class TestCumprodDtypeFloat64(TestCumprod):
    def init_dtype(self):
        self.dtype = np.float32
        self.val_dtype = np.float32

    @skip_if_not_cpu_or_gpu
    def test_dtype_float64(self):
        self.prepare_test_data(dim=1, zero_num=0)

        x = paddle.to_tensor(self.x, dtype='float32')
        x.stop_gradient = False
        out = paddle.cumprod(x, dim=1, dtype='float64')
        self.assertEqual(out.dtype, paddle.float64)

        out_ref = np.cumprod(self.x.astype(np.float32), axis=1).astype(
            np.float64
        )
        np.testing.assert_allclose(out_ref, out.numpy(), rtol=1e-05)

        loss = paddle.sum(out)
        loss.backward()
        self.assertEqual(x.grad.dtype, paddle.float32)

        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)


class TestCumprodDtypeStatic(unittest.TestCase):
    def setUp(self):
        self.shape = [2, 3, 4]
        self.x = (np.random.rand(*self.shape) + 0.5).astype(np.float32)
        self.places = get_places()

    @skip_if_not_cpu_or_gpu
    def test_static_dtype_float32(self):
        paddle.enable_static()
        for place in self.places:
            with paddle.static.program_guard(paddle.static.Program()):
                x = paddle.static.data('X', self.shape, dtype='float32')
                out = paddle.cumprod(x, dim=1, dtype='float32')
                exe = paddle.static.Executor(place)
                (out_res,) = exe.run(feed={'X': self.x}, fetch_list=[out])

                out_ref = np.cumprod(self.x, axis=1).astype(np.float32)
                np.testing.assert_allclose(out_ref, out_res, rtol=1e-05)


class TestCumprodBoundaryConditions(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()

    def tearDown(self):
        paddle.enable_static()

    @skip_if_not_cpu_or_gpu
    def test_single_element_tensor(self):
        x = paddle.to_tensor([5.0], dtype='float32', stop_gradient=False)
        out = paddle.cumprod(x, dim=0)

        self.assertEqual(out.shape, [1])
        np.testing.assert_allclose(out.numpy(), [5.0], rtol=1e-05)

        out.backward()
        np.testing.assert_allclose(x.grad.numpy(), [1.0], rtol=1e-05)

    @skip_if_not_cpu_or_gpu
    def test_zero_values_gradient(self):
        x_data = np.array([[1.0, 0.0, 3.0], [2.0, 4.0, 0.0]], dtype=np.float32)
        x = paddle.to_tensor(x_data, stop_gradient=False)

        out = paddle.cumprod(x, dim=1)
        loss = paddle.sum(out)
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)

    @skip_if_not_cpu_or_gpu
    def test_negative_dim(self):
        x_data = np.random.rand(2, 3, 4).astype(np.float32) + 0.5
        x = paddle.to_tensor(x_data, stop_gradient=False)

        out1 = paddle.cumprod(x, dim=-1)
        out2 = paddle.cumprod(x, dim=2)

        np.testing.assert_allclose(out1.numpy(), out2.numpy(), rtol=1e-05)

        loss1 = paddle.sum(out1)
        loss1.backward()

        self.assertIsNotNone(x.grad)


if __name__ == "__main__":
    unittest.main()
