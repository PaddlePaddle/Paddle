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

import paddle
from paddle.incubate.nn.functional import fast_ln, fast_rms_norm

# Suppress eager mode warnings
paddle.disable_static()


class TestFastNorm(unittest.TestCase):
    """
    Tests the correctness of forward and backward propagation for fast_ln and fast_rms_norm.
    """

    def _fast_ln_ref(self, x_in, scale_in, bias_in, epsilon):
        """
        High-precision (float64) reference implementation for LayerNorm.
        """
        x = paddle.cast(x_in, 'float64')
        scale = paddle.cast(scale_in, 'float64')
        bias = paddle.cast(bias_in, 'float64')
        mean = paddle.mean(x, axis=-1, keepdim=True)
        variance = paddle.mean(paddle.square(x - mean), axis=-1, keepdim=True)
        invvar = paddle.rsqrt(variance + epsilon)
        y = (x - mean) * invvar
        y = y * scale + bias
        return y.astype(x_in.dtype), mean, invvar

    def _fast_rms_ref(self, x_in, scale_in, epsilon):
        """
        High-precision (float64) reference implementation for RMSNorm.
        """
        x = paddle.cast(x_in, 'float64')
        scale = paddle.cast(scale_in, 'float64')
        variance = paddle.mean(paddle.square(x), axis=-1, keepdim=True)
        invvar = paddle.rsqrt(variance + epsilon)
        y = x * invvar
        y = y * scale
        return y.astype(x_in.dtype), invvar

    def _assert_allclose(self, a, b, atol, rtol, msg=""):
        """
        Custom assertion to report maximum absolute and relative errors.
        """
        a_f32 = a.astype('float32')
        b_f32 = b.astype('float32')
        abs_error = paddle.abs(a_f32 - b_f32)
        max_abs_error = paddle.max(abs_error).item()

        # Avoid division by zero
        rel_error = abs_error / (paddle.abs(b_f32) + 1e-9)
        max_rel_error = paddle.max(rel_error).item()

        if max_rel_error > rtol or max_abs_error > atol:
            self.fail(
                f"{msg} - Verification failed! "
                f"Max absolute error: {max_abs_error:.6e} (Tolerance: {atol:.6e}), "
                f"Max relative error: {max_rel_error:.6e} (Tolerance: {rtol:.6e})"
            )

    def test_fast_ln_forward_backward(self):
        """
        Tests the forward and gradient correctness of fast_ln.
        """
        paddle.seed(114514)

        params = [
            (1, 8192, 1024, "float32", 1e-4),
            (1, 8192, 1024, "float16", 1e-2),
            (1, 8192, 1024, "bfloat16", 1e-1),
        ]

        fixed_rtol = 1.0

        for B, C, H, dtype, atol in params:
            with self.subTest(shape=(B, C, H), dtype=dtype):
                # 1. Initialize inputs
                shape = [B, C, H]
                x_ref = paddle.randn(shape=shape, dtype=dtype)
                x_proposed = x_ref.clone()
                x_ref.stop_gradient = False
                x_proposed.stop_gradient = False

                scale_init = paddle.ones(shape=[H], dtype=dtype)
                bias_init = paddle.zeros(shape=[H], dtype=dtype)

                scale_ref = scale_init.clone()
                scale_proposed = scale_init.clone()
                bias_ref = bias_init.clone()
                bias_proposed = bias_init.clone()

                scale_ref.stop_gradient = False
                scale_proposed.stop_gradient = False
                bias_ref.stop_gradient = False
                bias_proposed.stop_gradient = False

                epsilon = 1e-5

                # 2. Forward computation
                y_ref, _, _ = self._fast_ln_ref(
                    x_ref, scale_ref, bias_ref, epsilon=epsilon
                )
                y_proposed, _, _ = fast_ln(
                    x_proposed, scale_proposed, bias_proposed, epsilon=epsilon
                )

                # 3. Gradient computation
                y_ref.sum().backward()
                y_proposed.sum().backward()

                # 4. Verification (Forward)
                self._assert_allclose(
                    y_ref,
                    y_proposed,
                    atol=atol,
                    rtol=fixed_rtol,
                    msg=f"fast_ln forward failed, dtype={dtype}",
                )

                # 5. Verification (Gradient)
                self._assert_allclose(
                    x_ref.grad,
                    x_proposed.grad,
                    atol=atol,
                    rtol=fixed_rtol,
                    msg=f"fast_ln input gradient failed, dtype={dtype}",
                )
                self._assert_allclose(
                    scale_ref.grad,
                    scale_proposed.grad,
                    atol=atol,
                    rtol=fixed_rtol,
                    msg=f"fast_ln Scale gradient failed, dtype={dtype}",
                )
                self._assert_allclose(
                    bias_ref.grad,
                    bias_proposed.grad,
                    atol=atol,
                    rtol=fixed_rtol,
                    msg=f"fast_ln Bias gradient failed, dtype={dtype}",
                )

    def test_fast_rms_norm_forward_backward(self):
        """
        Tests the forward and gradient correctness of fast_rms_norm.
        """
        paddle.seed(114514)

        # Parameter list: (B, C, H, dtype, rtol)
        params = [
            (1, 8192, 1024, "float32", 2e-4),
            (1, 8192, 1024, "bfloat16", 1.5e-2),
        ]

        fixed_atol = 1.0

        for B, C, H, dtype, rtol in params:
            with self.subTest(shape=(B, C, H), dtype=dtype):
                # 1. Initialize inputs
                shape = [B, C, H]
                x_ref = paddle.randn(shape=shape, dtype=dtype)
                x_proposed = x_ref.clone()
                x_ref.stop_gradient = False
                x_proposed.stop_gradient = False

                scale_init = paddle.ones(shape=[H], dtype=dtype)
                scale_ref = scale_init.clone()
                scale_proposed = scale_init.clone()

                scale_ref.stop_gradient = False
                scale_proposed.stop_gradient = False

                epsilon = 1e-5

                # 2. Forward computation
                y_ref, _ = self._fast_rms_ref(x_ref, scale_ref, epsilon=epsilon)
                y_proposed, _ = fast_rms_norm(
                    x_proposed, scale_proposed, epsilon=epsilon
                )

                # 3. Gradient computation
                y_ref.sum().backward()
                y_proposed.sum().backward()

                # 4. Verification (Forward)
                self._assert_allclose(
                    y_ref,
                    y_proposed,
                    atol=fixed_atol,
                    rtol=rtol,
                    msg=f"fast_rms_norm forward failed, dtype={dtype}",
                )

                # 5. Verification (Gradient)
                self._assert_allclose(
                    x_ref.grad,
                    x_proposed.grad,
                    atol=fixed_atol,
                    rtol=rtol,
                    msg=f"fast_rms_norm input gradient failed, dtype={dtype}",
                )
                self._assert_allclose(
                    scale_ref.grad,
                    scale_proposed.grad,
                    atol=fixed_atol,
                    rtol=rtol,
                    msg=f"fast_rms_norm Scale gradient failed, dtype={dtype}",
                )


if __name__ == "__main__":
    unittest.main()
