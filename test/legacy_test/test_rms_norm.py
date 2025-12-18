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

import numpy as np

import paddle
from paddle.nn.functional import rms_norm_nzs


class TestRMSNorm(unittest.TestCase):
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

    def test_2d_input(self):
        rows, cols = 32, 64
        x = paddle.randn([rows, cols])
        scale = paddle.randn([cols])

        y_fused, invvar_fused = rms_norm_nzs(x, (cols,), scale)

        y_ref, invvar_ref = self.rms_norm_reference(x, scale)

        np.testing.assert_allclose(y_fused, y_ref, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(
            invvar_fused, invvar_ref, rtol=1e-5, atol=1e-5
        )

    def test_3d_input(self):
        batch, rows, cols = 16, 32, 64
        x = paddle.randn([batch, rows, cols])
        scale = paddle.randn([cols])

        y_fused, invvar_fused = rms_norm_nzs(x, (cols,), scale)

        y_ref, invvar_ref = self.rms_norm_reference(x, scale)

        np.testing.assert_allclose(
            y_fused.astype("float32"),
            y_ref.astype("float32"),
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            invvar_fused, invvar_ref, rtol=1e-5, atol=1e-5
        )

    def test_without_bias(self):
        rows, cols = 32, 64
        x = paddle.randn([rows, cols])
        scale = paddle.randn([cols])

        y_fused, invvar_fused = rms_norm_nzs(x, (cols,), scale)

        y_ref, invvar_ref = self.rms_norm_reference(x, scale)

        np.testing.assert_allclose(y_fused, y_ref, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(
            invvar_fused, invvar_ref, rtol=1e-5, atol=1e-5
        )

    def test_3d_backward(self):
        batch, rows, cols = 8, 16, 32
        x = paddle.randn([batch, rows, cols], dtype='float32')
        x.stop_gradient = False
        scale = paddle.randn([cols], dtype='float32')
        scale.stop_gradient = False

        y_fused, invvar = rms_norm_nzs(x, (cols,), scale)

        loss = paddle.mean(y_fused)
        loss.backward()

        x_grad_fused = x.grad.clone()
        scale_grad_fused = scale.grad.clone()

        x.clear_gradient()
        scale.clear_gradient()

        y_ref, invvar_ref = self.rms_norm_reference(x, scale)
        loss_ref = paddle.mean(y_ref)
        loss_ref.backward()

        x_grad_ref = x.grad
        scale_grad_ref = scale.grad

        np.testing.assert_allclose(
            x_grad_fused, x_grad_ref, rtol=1e-4, atol=1e-4
        )
        np.testing.assert_allclose(
            scale_grad_fused, scale_grad_ref, rtol=1e-4, atol=1e-4
        )

    def test_backward(self):
        rows, cols = 16, 32
        test_type = ['bfloat16', 'float32']
        for x_type in test_type:
            for scale_type in test_type:
                x = paddle.randn([rows, cols], dtype=x_type)
                x.stop_gradient = False
                scale = paddle.randn([cols], dtype=scale_type)
                scale.stop_gradient = False

                y_fused, invvar = rms_norm_nzs(x, (cols,), scale)

                loss = paddle.mean(y_fused)
                loss.backward()

                x_grad_fused = x.grad.clone()
                scale_grad_fused = scale.grad.clone()

                x.clear_gradient()
                scale.clear_gradient()

                y_ref, invvar_ref = self.rms_norm_reference(x, scale)
                loss_ref = paddle.mean(y_ref)
                loss_ref.backward()

                x_grad_ref = x.grad
                scale_grad_ref = scale.grad

                np.testing.assert_allclose(
                    x_grad_fused.astype("float32"),
                    x_grad_ref.astype("float32"),
                    rtol=1e-4,
                    atol=1e-4,
                )
                np.testing.assert_allclose(
                    scale_grad_fused.astype("float32"),
                    scale_grad_ref.astype("float32"),
                    rtol=1e-4,
                    atol=1e-4,
                )


class TestFastRMSNorm(unittest.TestCase):
    """
    Tests the correctness of forward and backward propagation for rms_norm_nzs.
    """

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

    def test_fast_rms_norm_forward_backward(self):
        """
        Tests the forward and gradient correctness of rms_norm_nzs.
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
                y_proposed, _ = rms_norm_nzs(
                    x_proposed, (H,), scale_proposed, epsilon=epsilon
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
                    msg=f"rms_norm_nzs forward failed, dtype={dtype}",
                )

                # 5. Verification (Gradient)
                self._assert_allclose(
                    x_ref.grad,
                    x_proposed.grad,
                    atol=fixed_atol,
                    rtol=rtol,
                    msg=f"rms_norm_nzs input gradient failed, dtype={dtype}",
                )
                self._assert_allclose(
                    scale_ref.grad,
                    scale_proposed.grad,
                    atol=fixed_atol,
                    rtol=rtol,
                    msg=f"rms_norm_nzs Scale gradient failed, dtype={dtype}",
                )


if __name__ == '__main__':
    unittest.main()
