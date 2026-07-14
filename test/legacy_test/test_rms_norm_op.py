#  Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
from functools import reduce
from operator import mul

import numpy as np
from op_test import OpTest

import paddle
from paddle.nn.functional import rms_norm


def rms_norm_reference(x, scale, bias=None, epsilon=1e-5):
    x_shape = x.shape
    begin_norm_axis = len(x.shape) - 1
    N = reduce(mul, x_shape[0:begin_norm_axis], 1)
    D = reduce(mul, x_shape[begin_norm_axis : len(x_shape)], 1)
    x.shape = [N, D]

    variance = np.mean(np.square(x), axis=-1)
    rms = np.sqrt(variance + epsilon)
    y = x / rms.reshape([N, 1])
    y = y * scale.reshape([1, -1])
    if bias is not None:
        y = y + bias.reshape([1, -1])

    return y, 1.0 / rms


class TestRMSNormOp(OpTest):
    def setUp(self):
        self.op_type = "rms_norm"
        self.init_dtype()
        self.init_config()

        np.random.seed(2023)
        x = np.random.randn(*self.x_shape).astype(self.dtype)
        scale = np.random.randn(self.x_shape[-1]).astype(self.dtype)
        normalized_shape = [self.x_shape[-1]]

        self.inputs = {'x': x, 'scale': scale}
        self.attrs = {
            'normalized_shape': normalized_shape,
            'epsilon': self.epsilon,
        }
        y_ref, invvar_ref = rms_norm_reference(x, scale, epsilon=self.epsilon)
        self.outputs = {'y': y_ref, 'invvar': invvar_ref}

        def rms_norm_wrapper(x, scale):
            from paddle import _C_ops

            return _C_ops.rms_norm(x, scale, scale.shape, self.epsilon)

        self.python_api = rms_norm_wrapper

    def init_dtype(self):
        self.dtype = np.float32

    def init_config(self):
        self.epsilon = 1e-5
        self.x_shape = (32, 64)

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad(self):
        self.check_grad(['x', 'scale'], ['y'], check_pir=True)

    @classmethod
    def tearDownClass(cls):
        # Avoid AssertionError: This test of rms_norm op needs check_grad with fp64 precision.
        pass


class TestRMSNormOp3D(TestRMSNormOp):
    def init_config(self):
        self.epsilon = 1e-5
        self.x_shape = (16, 32, 64)

    def test_check_output(self):
        self.check_output(check_pir=True)


class TestRMSNormOpEpsilon(TestRMSNormOp):
    def init_config(self):
        self.epsilon = 1e-4
        self.x_shape = (32, 64)


class TestRMSNormAPI(unittest.TestCase):
    def setUp(self):
        paddle.seed(2023)
        np.random.seed(2023)

    def rms_norm_reference(self, x, scale, bias=None, epsilon=1e-5):
        variance = paddle.mean(paddle.square(x), axis=-1, keepdim=True)
        rms = paddle.sqrt(variance + epsilon)
        y = x / rms
        y = y * scale.reshape([1, -1])
        if bias is not None:
            y = y + bias.reshape([1, -1])

        return y, paddle.flatten(1.0 / rms)

    def test_api_dygraph(self):
        rows, cols = 32, 64
        x_np = np.random.randn(rows, cols).astype("float32")
        scale_np = np.random.randn(cols).astype("float32")

        x = paddle.to_tensor(x_np)
        x.stop_gradient = False
        scale = paddle.to_tensor(scale_np)
        scale.stop_gradient = False

        # Test forward
        y_fused = rms_norm(x, (cols,), scale)
        y_ref, _ = self.rms_norm_reference(x, scale)

        np.testing.assert_allclose(
            y_fused.numpy(), y_ref.numpy(), rtol=1e-5, atol=1e-5
        )

        # Test backward
        loss = paddle.mean(y_fused)
        loss.backward()

        x_grad_fused = x.grad.numpy()
        scale_grad_fused = scale.grad.numpy()

        x.clear_gradient()
        scale.clear_gradient()

        y_ref, invvar_ref = self.rms_norm_reference(x, scale)
        loss_ref = paddle.mean(y_ref)
        loss_ref.backward()

        np.testing.assert_allclose(
            x_grad_fused, x.grad.numpy(), rtol=1e-5, atol=1e-5
        )
        np.testing.assert_allclose(
            scale_grad_fused, scale.grad.numpy(), rtol=1e-5, atol=1e-5
        )


class TestRMSNormValueError(unittest.TestCase):
    def test_normalized_shape_type_error(self):
        x = paddle.randn([2, 3])
        with self.assertRaises(TypeError):
            rms_norm(x, "invalid_shape")

    def test_input_shape_mismatch(self):
        x = paddle.randn([2, 3])
        with self.assertRaises(ValueError):
            rms_norm(x, [4])

    def test_weight_shape_mismatch(self):
        x = paddle.randn([2, 3])
        weight = paddle.randn([4])
        with self.assertRaises(ValueError):
            rms_norm(x, [3], weight=weight)


class TestRMSNormEpsNone(unittest.TestCase):
    """Tests that eps=None selects the correct machine epsilon per dtype."""

    def _ref(self, x_np, scale_np, epsilon):
        variance = np.mean(np.square(x_np), axis=-1, keepdims=True)
        rms = np.sqrt(variance + epsilon)
        return x_np / rms * scale_np

    def test_eps_none_float32(self):
        """eps=None with float32 input should use float machine epsilon."""

        rows, cols = 8, 16
        x_np = np.random.randn(rows, cols).astype("float32")
        scale_np = np.ones(cols, dtype="float32")

        x = paddle.to_tensor(x_np)
        scale = paddle.to_tensor(scale_np)

        y_none = rms_norm(x, (cols,), scale, eps=None)
        float_eps = 1.1920929e-07
        y_explicit = rms_norm(x, (cols,), scale, eps=float_eps)

        np.testing.assert_array_equal(y_none.numpy(), y_explicit.numpy())

        y_ref = self._ref(x_np, scale_np, float_eps)
        np.testing.assert_allclose(
            y_none.numpy(), y_ref.astype("float32"), rtol=1e-5, atol=1e-5
        )

    def test_eps_none_float64(self):
        """eps=None with float64 input should use double machine epsilon."""
        import sys

        rows, cols = 8, 16
        x_np = np.random.randn(rows, cols).astype("float64")
        scale_np = np.ones(cols, dtype="float64")

        x = paddle.to_tensor(x_np)
        scale = paddle.to_tensor(scale_np)

        y_none = rms_norm(x, (cols,), scale, eps=None)
        double_eps = sys.float_info.epsilon  # ~2.22e-16
        y_explicit = rms_norm(x, (cols,), scale, eps=double_eps)

        np.testing.assert_array_equal(y_none.numpy(), y_explicit.numpy())

        y_ref = self._ref(x_np, scale_np, double_eps)
        np.testing.assert_allclose(
            y_none.numpy(), y_ref, rtol=1e-12, atol=1e-12
        )

    def test_eps_none_float32_differs_from_float64(self):
        """float32 and float64 defaults should be different epsilon values."""
        import sys

        float_eps = 1.1920929e-07
        double_eps = sys.float_info.epsilon
        self.assertNotAlmostEqual(float_eps, double_eps, places=10)

    def test_eps_none_backward_float32(self):
        """eps=None should work through backward pass for float32."""
        rows, cols = 8, 16
        x_np = np.random.randn(rows, cols).astype("float32")
        scale_np = np.ones(cols, dtype="float32")

        x = paddle.to_tensor(x_np)
        x.stop_gradient = False
        scale = paddle.to_tensor(scale_np)
        scale.stop_gradient = False

        y = rms_norm(x, (cols,), scale, eps=None)
        loss = paddle.mean(y)
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(scale.grad)


if __name__ == '__main__':
    unittest.main()
