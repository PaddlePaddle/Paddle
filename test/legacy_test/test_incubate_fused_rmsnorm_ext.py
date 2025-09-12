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
from paddle.incubate.nn.functional import fused_rms_norm_ext


class TestFusedRMSNorm(unittest.TestCase):
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

        y_fused, invvar_fused = fused_rms_norm_ext(x, scale)

        y_ref, invvar_ref = self.rms_norm_reference(x, scale)

        np.testing.assert_allclose(y_fused, y_ref, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(
            invvar_fused, invvar_ref, rtol=1e-5, atol=1e-5
        )

    def test_3d_input(self):
        batch, rows, cols = 16, 32, 64
        x = paddle.randn([batch, rows, cols])
        scale = paddle.randn([cols])

        y_fused, invvar_fused = fused_rms_norm_ext(x, scale)

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

        y_fused, invvar_fused = fused_rms_norm_ext(x, scale)

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

        y_fused, invvar = fused_rms_norm_ext(x, scale)

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

                y_fused, invvar = fused_rms_norm_ext(x, scale)

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


if __name__ == '__main__':
    unittest.main()
