#  Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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


def naive_rmsnorm(
    x,
    gamma,
    beta=None,
    epsilon=1e-6,
    begin_norm_axis=1,
    bias=None,
    residual=None,
):
    if residual is not None:
        x = x + residual
    if bias is not None:
        x = x + bias
    variance = x.pow(2).mean(-1, keepdim=True)
    out = paddle.rsqrt(variance + epsilon) * x
    out = out * gamma
    if beta is not None:
        out = out + beta
    return out


def fused_rmsnorm(
    x,
    gamma,
    beta=None,
    epsilon=1e-6,
    begin_norm_axis=1,
    bias=None,
    residual=None,
):
    out = paddle.incubate.nn.functional.fused_rms_norm(
        x,
        gamma,
        beta,
        epsilon,
        begin_norm_axis,
        bias,
        residual,
    )
    return out[0]


def check_allclose(out1, out2, rtol, atol):
    if out1.dtype == paddle.bfloat16:
        out1 = out1.astype(paddle.float32)
        out2 = out2.astype(paddle.float32)
    np.testing.assert_allclose(
        out1.numpy(),
        out2.numpy(),
        rtol=rtol,
        atol=atol,
    )


class TestRMSNormOp(unittest.TestCase):
    def setUp(self):
        np.random.seed(20)
        paddle.seed(20)
        paddle.disable_static()
        self.set_dtype()
        self.set_shape()
        self.set_begin_norm_axis()
        self.set_epsilon()
        self.set_tolerance()
        self.set_data()

    def set_dtype(self):
        self.dtype = paddle.float32

    def set_shape(self):
        self.rows = 32
        self.cols = 256

    def set_begin_norm_axis(self):
        self.begin_norm_axis = 1

    def set_epsilon(self):
        self.epsilon = 1e-6

    def set_tolerance(self):
        self.rtol = 1e-5
        self.atol = 1e-5

    def set_data(self):
        self.x_np = np.random.random([self.rows, self.cols])
        self.norm_weight_np = np.random.random([self.cols])
        self.norm_bias_np = np.random.random([self.cols])
        self.residual_np = np.random.random([self.rows, self.cols])
        self.bias_np = np.random.random([self.cols])

        self.x = paddle.to_tensor(self.x_np).astype(self.dtype)
        self.norm_weight = paddle.to_tensor(self.norm_weight_np).astype(
            self.dtype
        )
        self.x.stop_gradient = False
        self.norm_weight.stop_gradient = False

        self.norm_bias = paddle.to_tensor(self.norm_bias_np).astype(self.dtype)
        self.residual = paddle.to_tensor(self.residual_np).astype(self.dtype)
        self.bias = paddle.to_tensor(self.bias_np).astype(self.dtype)

    def get_forward_output(self, func):
        out = func(
            self.x,
            self.norm_weight,
            self.norm_bias,
            self.epsilon,
            self.begin_norm_axis,
        )
        return out

    def get_residual_bias_forward_output(self, func):
        out = func(
            self.x,
            self.norm_weight,
            self.norm_bias,
            self.epsilon,
            self.begin_norm_axis,
            self.bias,
            self.residual,
        )
        return out

    def get_forward_backward_output(self, func):
        out = func(
            self.x,
            self.norm_weight,
            self.norm_bias,
            self.epsilon,
            self.begin_norm_axis,
        )
        out_grad = paddle.randn([self.rows, self.cols], self.dtype)
        paddle.autograd.backward([out], [out_grad], True)
        return out, (self.x.grad, self.norm_weight.grad)

    def test_rmsnorm_residual_bias_forward(self):
        naive_out = self.get_residual_bias_forward_output(naive_rmsnorm)
        fused_out = self.get_residual_bias_forward_output(fused_rmsnorm)
        check_allclose(naive_out, fused_out, self.rtol, self.atol)

    def test_rmsnorm_forward_backward(self):
        naive_out, naive_grads = self.get_forward_backward_output(naive_rmsnorm)
        fused_out, fused_grads = self.get_forward_backward_output(fused_rmsnorm)

        # check forward
        check_allclose(naive_out, fused_out, self.rtol, self.atol)

        # check backward
        naive_x_grad, naive_scale_grad = naive_grads
        fused_x_grad, fused_scale_grad = fused_grads
        check_allclose(naive_x_grad, fused_x_grad, self.rtol, self.atol)
        check_allclose(naive_scale_grad, fused_scale_grad, self.rtol, self.atol)


class TestRMSNormOp2(TestRMSNormOp):
    def set_dtype(self):
        self.dtype = paddle.float16

    def set_tolerance(self):
        self.rtol = 2e-3
        self.atol = 2e-3


class TestRMSNormOp3(TestRMSNormOp):
    def set_dtype(self):
        self.dtype = paddle.bfloat16

    def set_tolerance(self):
        self.rtol = 3e-2
        self.atol = 3e-2


class TestRMSNormOp4(TestRMSNormOp):
    def set_shape(self):
        self.rows = 1024
        self.cols = 2048


if __name__ == "__main__":
    unittest.main()
