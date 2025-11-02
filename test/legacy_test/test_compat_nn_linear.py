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
from paddle.compat.nn import Linear


class TestCompatLinearLayer(unittest.TestCase):
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

    def _create_linear_layer(
        self,
        in_features,
        out_features,
        bias=True,
        weight_np=None,
        bias_np=None,
        dtype=None,
    ):
        """Create Linear layer with specific weights"""
        linear = Linear(in_features, out_features, bias=bias, dtype=dtype)

        # Set custom weights if provided
        if weight_np is not None:
            linear.weight.set_value(paddle.to_tensor(weight_np))

        if bias and bias_np is not None:
            linear.bias.set_value(paddle.to_tensor(bias_np))

        return linear

    def _compare_forward(self, x_np, weight_np, bias_np=None, dtype=None):
        """Compare forward pass with NumPy implementation"""
        # NumPy calculation
        y_np = self._numpy_linear_forward(x_np, weight_np, bias_np)

        # Paddle calculation with Linear layer
        in_features = weight_np.shape[1]
        out_features = weight_np.shape[0]

        linear = self._create_linear_layer(
            in_features,
            out_features,
            bias=(bias_np is not None),
            weight_np=weight_np,
            bias_np=bias_np,
            dtype=dtype,
        )

        x_pd = paddle.to_tensor(x_np)
        y_pd = linear(x_pd)

        # Compare results
        rtol, atol = self.get_error_range(is_large=x_np.size > 8192)
        np.testing.assert_allclose(y_pd.numpy(), y_np, rtol=rtol, atol=atol)

    def _compare_backward(self, x_np, weight_np, bias_np=None, dtype=None):
        """Compare backward pass with NumPy implementation"""
        in_features = weight_np.shape[1]
        out_features = weight_np.shape[0]

        # Create Linear layer with custom weights
        linear = self._create_linear_layer(
            in_features,
            out_features,
            bias=(bias_np is not None),
            weight_np=weight_np,
            bias_np=bias_np,
            dtype=dtype,
        )

        # Prepare input tensor
        x_pd = paddle.to_tensor(x_np, stop_gradient=False)

        # Forward pass
        y_pd = linear(x_pd)

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
            linear.weight.grad.numpy(), dw_np, rtol=rtol, atol=atol
        )
        if bias_np is not None:
            np.testing.assert_allclose(
                linear.bias.grad.numpy(), db_np, rtol=rtol, atol=atol
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
        dtypes = ["float32", "float64"]
        if paddle.base.is_compiled_with_cuda():
            dtypes.append("float16")

        for dtype in dtypes:
            x_np = np.random.randn(4, 3).astype(dtype)
            weight_np = np.random.randn(5, 3).astype(dtype)
            bias_np = np.random.randn(5).astype(dtype)

            self._compare_forward(x_np, weight_np, bias_np, dtype)
            self._compare_backward(x_np, weight_np, bias_np, dtype)

    def test_static_graph_simple(self):
        """Test Linear layer in static graph mode"""
        if not paddle.base.is_compiled_with_cuda():
            return
        paddle.enable_static()

        try:
            program = paddle.static.Program()
            startup_program = paddle.static.Program()

            with paddle.static.program_guard(program, startup_program):
                # Create input data
                x = paddle.static.data(name='x', shape=[2, 3], dtype='float32')

                # Create Linear layer (let it initialize its own weights)
                linear = Linear(3, 4, bias=True)
                y = linear(x)

                # Get weight and bias tensors for GT calculation
                weight = linear.weight
                bias = linear.bias

                place = paddle.CUDAPlace(0)
                exe = paddle.static.Executor(place)
                exe.run(startup_program)

                # Simple deterministic input
                x_np = np.ones([2, 3], dtype=np.float32)

                # Run and get results including weight and bias
                results = exe.run(
                    feed={'x': x_np}, fetch_list=[y, weight, bias]
                )

                y_pd, weight_np, bias_np = results

                # Calculate GT using numpy with the actual weights from Linear layer
                y_gt = self._numpy_linear_forward(x_np, weight_np, bias_np)

                # Compare results
                np.testing.assert_allclose(y_pd, y_gt, rtol=1e-5, atol=1e-8)
        finally:
            paddle.disable_static()

    def test_device_and_dtype_parameters(self):
        """Test device and dtype parameters"""
        # Test CPU device
        linear_cpu = Linear(3, 5, device='cpu', dtype='float32')
        self.assertEqual(linear_cpu.weight.place.is_cpu_place(), True)
        self.assertEqual(linear_cpu.weight.dtype, paddle.float32)

        # if paddle.is_compiled_with_cuda():
        #     # Test GPU device if available
        #     linear_gpu = Linear(3, 5, device='gpu', dtype='float32')
        #     self.assertEqual(linear_gpu.weight.place.is_gpu_place(), True)

        # Test different dtype
        linear_fp64 = Linear(3, 5, dtype='float64')
        self.assertEqual(linear_fp64.weight.dtype, paddle.float64)

    def test_weight_initialization(self):
        """Test weight and bias initialization"""
        # Test default initialization
        linear = Linear(10, 20)

        # Check shape
        self.assertEqual(linear.weight.shape, [20, 10])
        self.assertEqual(linear.bias.shape, [20])

        # Check that weights are not all zeros
        self.assertFalse(np.allclose(linear.weight.numpy(), np.zeros((20, 10))))

        # Test without bias
        linear_no_bias = Linear(10, 20, bias=False)
        self.assertIsNone(linear_no_bias.bias)

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

        # Paddle calculation with Linear layer
        linear = self._create_linear_layer(
            2, 2, weight_np=weight_np, bias=False
        )
        x_pd = paddle.to_tensor(x_np)
        y_pd = linear(x_pd)

        np.testing.assert_allclose(y_pd.numpy(), expected, rtol=1e-5, atol=1e-8)

    def test_error_handling(self):
        """Test error handling for invalid inputs"""
        # Shape mismatch between input and weight
        with self.assertRaises(ValueError):
            linear = Linear(3, 5)
            x = paddle.to_tensor(
                np.random.randn(3, 4).astype(np.float32)
            )  # Last dim should be 3
            linear(x)

        wrong_api_used = (
            "paddle{module}.nn.Linear() received unexpected keyword argument{plural} {args}. "
            "\nDid you mean to use paddle{correct_module}.nn.Linear() instead?"
        )

        with self.assertRaises(TypeError) as cm:
            lin = paddle.compat.nn.Linear(
                3,
                5,
                weight_attr=None,
                name='linear_layer',
            )
        self.assertEqual(
            str(cm.exception),
            wrong_api_used.format(
                module=".compat",
                args="'name', 'weight_attr'",
                correct_module="",
                plural="s",
            ),
        )

        with self.assertRaises(TypeError) as cm:
            lin = paddle.nn.Linear(
                3, 5, bias=True, device="cpu", dtype="float32"
            )
        self.assertEqual(
            str(cm.exception),
            wrong_api_used.format(
                module="",
                args="'bias', 'device', 'dtype'",
                correct_module=".compat",
                plural="s",
            ),
        )

    def test_state_dict(self):
        """Test state dict functionality"""
        linear = Linear(10, 20)

        # Get state dict
        state_dict = linear.state_dict()

        # Check keys
        self.assertIn('weight', state_dict)
        self.assertIn('bias', state_dict)

        # Create new linear and load state
        new_linear = Linear(10, 20)
        new_linear.set_state_dict(state_dict)

        # Check if weights are the same
        np.testing.assert_allclose(
            linear.weight.numpy(),
            new_linear.weight.numpy(),
            rtol=1e-5,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            linear.bias.numpy(), new_linear.bias.numpy(), rtol=1e-5, atol=1e-8
        )

    def test_parameters_method(self):
        """Test parameters() method"""
        linear = Linear(10, 20)

        # Get parameters
        params = list(linear.parameters())

        # Should return weight and bias
        self.assertEqual(len(params), 2)
        self.assertEqual(params[0].shape, [20, 10])
        self.assertEqual(params[1].shape, [20])

        # Test without bias
        linear_no_bias = Linear(10, 20, bias=False)
        params_no_bias = list(linear_no_bias.parameters())
        self.assertEqual(len(params_no_bias), 1)  # Only weight

    def test_train_eval_mode(self):
        """Test train and eval mode"""
        linear = Linear(10, 20)

        # Default should be train mode
        self.assertTrue(linear.training)

        # Switch to eval mode
        linear.eval()
        self.assertFalse(linear.training)

        # Switch back to train mode
        linear.train()
        self.assertTrue(linear.training)


if __name__ == "__main__":
    unittest.main()
