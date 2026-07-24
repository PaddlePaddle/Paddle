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

import unittest

import numpy as np

import paddle
from paddle.compat.nn import functional as F


class TestCompatLinear(unittest.TestCase):
    def setUp(self):
        self.seed = 42
        np.random.seed(self.seed)
        paddle.seed(self.seed)

    def get_error_range(self, is_large=False):
        # xpu matmul precision is very low, rtol cannot be set
        if paddle.core.is_compiled_with_xpu():
            return (0, 0.1)
        if is_large:
            return (1e-1, 1e-4)
        return (1e-3, 1e-6)

    def _numpy_linear_forward(self, x, weight, bias=None):
        """NumPy implementation of linear forward pass"""
        # Torch linear: y = x @ weight.T + bias
        # So we need to transpose weight for NumPy implementation
        result = np.dot(x, weight.T)
        if bias is not None:
            result += bias
        return result

    def _numpy_linear_backward(self, x, weight, bias, dy):
        """NumPy implementation of linear backward pass"""
        x_shape = x.shape
        dy_shape = dy.shape

        # Reshape to 2D: (batch_size * other_dims, in_features)
        x_2d = x.reshape(-1, x_shape[-1])
        dy_2d = dy.reshape(-1, dy_shape[-1])

        # dx = dy @ weight
        dx_2d = np.dot(dy_2d, weight)
        # dw = dy.T @ x
        dw = np.dot(dy_2d.T, x_2d)
        # db = sum(dy, axis=all_but_last)
        db = np.sum(dy_2d, axis=0) if bias is not None else None

        # Reshape dx back to original input shape (except last dimension)
        dx = dx_2d.reshape(*x_shape[:-1], dx_2d.shape[-1])

        return dx, dw, db

    def _compare_forward(self, x_np, weight_np, bias_np=None):
        """Compare forward pass with NumPy implementation"""
        # NumPy calculation
        y_np = self._numpy_linear_forward(x_np, weight_np, bias_np)

        # Paddle calculation
        x_pd = paddle.to_tensor(x_np)
        weight_pd = paddle.to_tensor(weight_np)
        bias_pd = paddle.to_tensor(bias_np) if bias_np is not None else None

        y_pd = paddle.compat.nn.functional.linear(x_pd, weight_pd, bias_pd)

        # Compare results
        rtol, atol = self.get_error_range(is_large=x_np.size > 8192)
        np.testing.assert_allclose(y_pd.numpy(), y_np, rtol=rtol, atol=atol)

    def _compare_backward(self, x_np, weight_np, bias_np=None):
        """Compare backward pass with NumPy implementation"""
        # Prepare Paddle tensors with gradients
        x_pd = paddle.to_tensor(x_np, stop_gradient=False)
        weight_pd = paddle.to_tensor(weight_np, stop_gradient=False)
        bias_pd = (
            paddle.to_tensor(bias_np, stop_gradient=False)
            if bias_np is not None
            else None
        )

        # Forward pass
        y_pd = paddle.compat.nn.functional.linear(x_pd, weight_pd, bias_pd)

        # Create upstream gradient (same shape as output)
        dy_np = np.random.randn(*y_pd.shape).astype(x_np.dtype)
        dy_pd = paddle.to_tensor(dy_np)

        # Backward pass
        y_pd.backward(dy_pd)

        # NumPy gradients
        dx_np, dw_np, db_np = self._numpy_linear_backward(
            x_np, weight_np, bias_np, dy_np
        )

        rtol, atol = self.get_error_range(is_large=x_np.size > 8192)
        # Compare gradients
        np.testing.assert_allclose(
            x_pd.grad.numpy(), dx_np, rtol=rtol, atol=atol
        )
        np.testing.assert_allclose(
            weight_pd.grad.numpy(), dw_np, rtol=rtol, atol=atol
        )
        if bias_np is not None:
            np.testing.assert_allclose(
                bias_pd.grad.numpy(), db_np, rtol=rtol, atol=atol
            )

    def test_2d_input_with_bias(self):
        """Test 2D input with bias"""
        x_np = np.random.randn(4, 3).astype(np.float32)
        weight_np = np.random.randn(5, 3).astype(np.float32)
        bias_np = np.random.randn(5).astype(np.float32)

        self._compare_forward(x_np, weight_np, bias_np)
        self._compare_backward(x_np, weight_np, bias_np)

    def test_2d_input_no_bias(self):
        """Test 2D input without bias"""
        x_np = np.random.randn(4, 3).astype(np.float32)
        weight_np = np.random.randn(5, 3).astype(np.float32)

        self._compare_forward(x_np, weight_np, None)
        self._compare_backward(x_np, weight_np, None)

    def test_1d_input_with_bias(self):
        """Test 1D input (no batch dimension) with bias"""
        x_np = np.random.randn(3).astype(np.float32)
        weight_np = np.random.randn(5, 3).astype(np.float32)
        bias_np = np.random.randn(5).astype(np.float32)

        self._compare_forward(x_np, weight_np, bias_np)
        self._compare_backward(x_np, weight_np, bias_np)

    def test_3d_input_with_bias(self):
        """Test 3D input with bias"""
        x_np = np.random.randn(2, 4, 3).astype(np.float32)
        weight_np = np.random.randn(5, 3).astype(np.float32)
        bias_np = np.random.randn(5).astype(np.float32)

        self._compare_forward(x_np, weight_np, bias_np)
        self._compare_backward(x_np, weight_np, bias_np)

    def test_4d_input_no_bias(self):
        """Test 4D input without bias"""
        x_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
        weight_np = np.random.randn(6, 5).astype(np.float32)

        self._compare_forward(x_np, weight_np, None)
        self._compare_backward(x_np, weight_np, None)

    def test_large_input_with_bias(self):
        """Test large input dimensions with bias"""
        x_np = np.random.randn(128, 512).astype(np.float32)
        weight_np = np.random.randn(256, 512).astype(np.float32)
        bias_np = np.random.randn(256).astype(np.float32)

        self._compare_forward(x_np, weight_np, bias_np)
        self._compare_backward(x_np, weight_np, bias_np)

    def test_non_contiguous_shapes(self):
        """Test non-power-of-two shapes"""
        x_np = np.random.randn(31, 63).astype(np.float32)
        weight_np = np.random.randn(127, 63).astype(np.float32)
        bias_np = np.random.randn(127).astype(np.float32)

        self._compare_forward(x_np, weight_np, bias_np)
        self._compare_backward(x_np, weight_np, bias_np)

    def test_different_dtypes(self):
        """Test different data types"""
        dtypes = [np.float32, np.float64]

        for dtype in dtypes:
            x_np = np.random.randn(4, 3).astype(dtype)
            weight_np = np.random.randn(5, 3).astype(dtype)
            bias_np = np.random.randn(5).astype(dtype)

            self._compare_forward(x_np, weight_np, bias_np)
            self._compare_backward(x_np, weight_np, bias_np)

    def test_static_graph_simple(self):
        if not paddle.base.is_compiled_with_cuda():
            return
        paddle.enable_static()

        try:
            # Simple fixed case
            program = paddle.static.Program()
            with paddle.static.program_guard(program):
                x = paddle.static.data(name='x', shape=[2, 3], dtype='float32')
                weight = paddle.full(
                    shape=[4, 3], fill_value=0.5, dtype='float32'
                )
                bias = paddle.ones(shape=[4], dtype='float32')

                y = paddle.compat.nn.functional.linear(x, weight, bias)

                place = paddle.CUDAPlace(0)
                exe = paddle.static.Executor(place)
                exe.run(paddle.static.default_startup_program())

                # Simple deterministic input
                x_np = np.ones([2, 3], dtype=np.float32)
                result = exe.run(feed={'x': x_np}, fetch_list=[y])[0]

                # Simple verification
                expected = np.array(
                    [[2.5, 2.5, 2.5, 2.5], [2.5, 2.5, 2.5, 2.5]],
                    dtype=np.float32,
                )
                np.testing.assert_allclose(result, expected, rtol=1e-5)

        finally:
            paddle.disable_static()

    def test_edge_cases(self):
        """Test edge cases"""
        # Empty input
        x_np = np.array([]).reshape(0, 3).astype(np.float32)
        weight_np = np.random.randn(5, 3).astype(np.float32)
        bias_np = np.random.randn(5).astype(np.float32)

        self._compare_forward(x_np, weight_np, bias_np)

        # Single element
        x_np = np.random.randn(1, 1).astype(np.float32)
        weight_np = np.random.randn(1, 1).astype(np.float32)
        bias_np = np.random.randn(1).astype(np.float32)

        self._compare_forward(x_np, weight_np, bias_np)
        self._compare_backward(x_np, weight_np, bias_np)

    def test_weight_transpose_behavior(self):
        """Test that weight is properly transposed (torch compatibility)"""
        # Create simple test case where transposition is obvious
        x_np = np.array([[1.0, 2.0]]).astype(np.float32)  # [1, 2]
        weight_np = np.array([[3.0, 4.0], [5.0, 6.0]]).astype(
            np.float32
        )  # [2, 2]

        # Manual calculation: x @ weight.T
        expected = np.array([[1 * 3 + 2 * 4, 1 * 5 + 2 * 6]]).astype(
            np.float32
        )  # [1, 2]

        # Paddle calculation
        x_pd = paddle.to_tensor(x_np)
        weight_pd = paddle.to_tensor(weight_np)
        y_pd = paddle.compat.nn.functional.linear(x_pd, weight_pd)

        np.testing.assert_allclose(y_pd.numpy(), expected, rtol=1e-5, atol=1e-8)

    def test_error_handling(self):
        """Test error handling for invalid inputs"""
        # Invalid weight shape (should be 2D)
        with self.assertRaises(ValueError):
            x = paddle.to_tensor(np.random.randn(3, 4).astype(np.float32))
            weight = paddle.to_tensor(
                np.random.randn(3).astype(np.float32)
            )  # 1D weight
            paddle.compat.nn.functional.linear(x, weight)

        # Shape mismatch
        with self.assertRaises(ValueError):
            x = paddle.to_tensor(np.random.randn(3, 4).astype(np.float32))
            weight = paddle.to_tensor(
                np.random.randn(5, 6).astype(np.float32)
            )  # Incompatible shapes
            paddle.compat.nn.functional.linear(x, weight)

        wrong_api_used = (
            "paddle{module}.nn.functional.linear() received unexpected keyword argument{plural} {args}. "
            "\nDid you mean to use paddle{correct_module}.nn.functional.linear() instead?"
        )

        with self.assertRaises(TypeError) as cm:
            tensors = F.linear(
                x=paddle.to_tensor([1, 2]),
                weight=paddle.to_tensor([[1, 2], [2, 1]]),
                bias=paddle.to_tensor([1, 1]),
                name='linear_layer',
            )
        self.assertEqual(
            str(cm.exception),
            wrong_api_used.format(
                module=".compat",
                args="'name', 'x'",
                correct_module="",
                plural="s",
            ),
        )


if __name__ == "__main__":
    unittest.main()
