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
from paddle import fluid
from paddle.fluid import core


def rmsnorm(x, norm_weight, norm_bias, epsilon):
    input_val = x
    out = paddle.incubate.nn.functional.rms_norm(
        input_val, norm_weight, norm_bias, epsilon, begin_norm_axis=1
    )
    return out


def residual_bias_add_rmsnorm(
    x, residual, bias, norm_weight, norm_bias, epsilon
):
    # Currently only layernorm support custom residual alpha.
    input_val = x
    input_val += residual
    input_val += bias
    residual_bias_add_input = input_val
    out = paddle.incubate.nn.functional.rms_norm(
        input_val, norm_weight, norm_bias, epsilon, begin_norm_axis=1
    )
    return out, residual_bias_add_input


def layernorm(x, norm_weight, norm_bias, epsilon):
    input_val = x
    norm_weight = paddle.cast(norm_weight, x.dtype)
    norm_bias = paddle.cast(norm_bias, x.dtype)
    out = paddle.nn.functional.layer_norm(
        input_val,
        x.shape[1:],
        weight=norm_weight,
        bias=norm_bias,
        epsilon=epsilon,
    )
    return out


def residual_bias_add_layernorm(
    x, residual, bias, norm_weight, norm_bias, epsilon, residual_alpha=1.0
):
    # Currently only layernorm support custom residual alpha.
    input_val = x
    input_val += bias
    input_val += residual * residual_alpha
    residual_bias_add_input = input_val
    norm_weight = paddle.cast(norm_weight, x.dtype)
    norm_bias = paddle.cast(norm_bias, x.dtype)
    out = paddle.nn.functional.layer_norm(
        input_val,
        x.shape[1:],
        weight=norm_weight,
        bias=norm_bias,
        epsilon=epsilon,
    )
    return out, residual_bias_add_input


# @unittest.skipIf(
#     not core.is_compiled_with_cuda(), "core is not compiled with CUDA "
# )
# class TestNormHelperOp(unittest.TestCase):
#     def setUp(self):
#         np.random.seed(20)
#         batch = 16
#         cols = 256
#         self.x_np = np.random.uniform(-0.05, 0.05, [batch, cols])
#         self.residual_np = np.random.uniform(-0.05, 0.05, [batch, cols])
#         self.bias_np = np.random.uniform(-0.05, 0.05, [cols])
#         self.gamma_np = np.random.uniform(-0.05, 0.05, [cols])
#         self.beta_np = np.random.uniform(-0.05, 0.05, [cols])
#         self.epsilon = 1e-5
#         self.residual_alpha = np.random.uniform(low=0.1, high=1.1, size=[1])

#     def check_layernorm(self, x_np, gamma_np, beta_np, dtype):
#         paddle.disable_static()
#         x = paddle.to_tensor(x_np.astype(dtype))
#         gamma = paddle.to_tensor(gamma_np.astype(dtype))
#         beta = paddle.to_tensor(beta_np.astype(dtype))

#         paddle_norm_helper_out = paddle.incubate.nn.functional.norm_helper(
#             x,
#             None,
#             None,
#             paddle.cast(gamma, paddle.float32),
#             paddle.cast(beta, paddle.float32),
#             self.epsilon,
#             1.0,
#             norm_type="layernorm",
#             begin_norm_axis=1,
#         )
#         paddle_naive_norm_out = layernorm(x, gamma, beta, self.epsilon)
#         paddle.enable_static()
#         return paddle_norm_helper_out, paddle_naive_norm_out

#     def check_bias_add_residual_layernorm(
#         self,
#         x_np,
#         residual_np,
#         bias_np,
#         gamma_np,
#         beta_np,
#         residual_alpha,
#         dtype,
#     ):
#         paddle.disable_static()
#         x = paddle.to_tensor(x_np.astype(dtype))
#         residual = paddle.to_tensor(residual_np.astype(dtype))
#         bias = paddle.to_tensor(bias_np.astype(dtype))
#         gamma = paddle.to_tensor(gamma_np.astype(dtype))
#         beta = paddle.to_tensor(beta_np.astype(dtype))

#         paddle_norm_helper_out = paddle.incubate.nn.functional.norm_helper(
#             x,
#             residual,
#             bias,
#             paddle.cast(gamma, paddle.float32),
#             paddle.cast(beta, paddle.float32),
#             self.epsilon,
#             residual_alpha,
#             norm_type="layernorm",
#             begin_norm_axis=1,
#         )
#         paddle_naive_norm_out = residual_bias_add_layernorm(
#             x, residual, bias, gamma, beta, self.epsilon, residual_alpha
#         )

#         paddle.enable_static()
#         return paddle_norm_helper_out, paddle_naive_norm_out

#     def check_rmsnorm(self, x_np, gamma_np, beta_np, dtype):
#         paddle.disable_static()
#         x = paddle.to_tensor(x_np.astype(dtype))
#         gamma = paddle.to_tensor(gamma_np.astype(dtype))
#         beta = paddle.to_tensor(beta_np.astype(dtype))

#         paddle_norm_helper_out = paddle.incubate.nn.functional.norm_helper(
#             x,
#             None,
#             None,
#             paddle.cast(gamma, x.dtype),
#             paddle.cast(beta, x.dtype),
#             self.epsilon,
#             1.0,
#             norm_type="rmsnorm",
#             begin_norm_axis=1,
#         )
#         paddle_naive_norm_out = rmsnorm(x, gamma, beta, self.epsilon)
#         paddle.enable_static()
#         return paddle_norm_helper_out, paddle_naive_norm_out

#     def check_bias_add_residual_rmsnorm(
#         self, x_np, residual_np, bias_np, gamma_np, beta_np, dtype
#     ):
#         paddle.disable_static()
#         x = paddle.to_tensor(x_np.astype(dtype))
#         residual = paddle.to_tensor(residual_np.astype(dtype))
#         bias = paddle.to_tensor(bias_np.astype(dtype))
#         gamma = paddle.to_tensor(gamma_np.astype(dtype))
#         beta = paddle.to_tensor(beta_np.astype(dtype))

#         paddle_norm_helper_out = paddle.incubate.nn.functional.norm_helper(
#             x,
#             residual,
#             bias,
#             paddle.cast(gamma, x.dtype),
#             paddle.cast(beta, x.dtype),
#             self.epsilon,
#             1.0,  # rmsnorm do not support custom residual alpha.
#             norm_type="rmsnorm",
#             begin_norm_axis=1,
#         )
#         paddle_naive_norm_out = residual_bias_add_rmsnorm(
#             x, residual, bias, gamma, beta, self.epsilon
#         )

#         paddle.enable_static()
#         return paddle_norm_helper_out, paddle_naive_norm_out

#     def test_norm_helper_fp16(self):
#         if not paddle.is_compiled_with_cuda():
#             return

#         paddle_norm_helper_rmsnorm, paddle_naive_rmsnorm = self.check_rmsnorm(
#             self.x_np, self.gamma_np, self.beta_np, 'float16'
#         )

#         np.testing.assert_allclose(
#             paddle_norm_helper_rmsnorm[0].numpy(),
#             paddle_naive_rmsnorm.numpy(),
#             rtol=1e-3,
#             atol=1e-3,
#         )

#         (
#             paddle_norm_helper_bias_add_residual_rmsnorm,
#             paddle_naive_bias_add_residual_rmsnorm,
#         ) = self.check_bias_add_residual_rmsnorm(
#             self.x_np,
#             self.residual_np,
#             self.bias_np,
#             self.gamma_np,
#             self.beta_np,
#             'float16',
#         )

#         np.testing.assert_allclose(
#             paddle_norm_helper_bias_add_residual_rmsnorm[0].numpy(),
#             paddle_naive_bias_add_residual_rmsnorm[0].numpy(),
#             rtol=1e-3,
#             atol=1e-3,
#         )

#         np.testing.assert_allclose(
#             paddle_norm_helper_bias_add_residual_rmsnorm[1].numpy(),
#             paddle_naive_bias_add_residual_rmsnorm[1].numpy(),
#             rtol=1e-3,
#             atol=1e-3,
#         )

#         (
#             paddle_norm_helper_layernorm,
#             paddle_naive_layernorm,
#         ) = self.check_layernorm(
#             self.x_np, self.gamma_np, self.beta_np, 'float16'
#         )

#         np.testing.assert_allclose(
#             paddle_norm_helper_layernorm[0].numpy(),
#             paddle_naive_layernorm.numpy(),
#             rtol=1e-3,
#             atol=1e-3,
#         )

#         (
#             paddle_norm_helper_bias_add_residual_layernorm,
#             paddle_naive_bias_add_residual_layernorm,
#         ) = self.check_bias_add_residual_layernorm(
#             self.x_np,
#             self.residual_np,
#             self.bias_np,
#             self.gamma_np,
#             self.beta_np,
#             self.residual_alpha[0],
#             'float16',
#         )

#         np.testing.assert_allclose(
#             paddle_norm_helper_bias_add_residual_layernorm[0].numpy(),
#             paddle_naive_bias_add_residual_layernorm[0].numpy(),
#             rtol=1e-3,
#             atol=1e-3,
#         )

#         np.testing.assert_allclose(
#             paddle_norm_helper_bias_add_residual_layernorm[1].numpy(),
#             paddle_naive_bias_add_residual_layernorm[1].numpy(),
#             rtol=1e-3,
#             atol=1e-3,
#         )


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA "
)
class TestNormHelperStaticOp(unittest.TestCase):
    def setUp(self):
        np.random.seed(20)
        self.batch = 16
        self.cols = 256
        self.x_np = np.random.uniform(-0.05, 0.05, [self.batch, self.cols])
        self.residual_np = np.random.uniform(
            -0.05, 0.05, [self.batch, self.cols]
        )
        self.bias_np = np.random.uniform(-0.05, 0.05, [self.cols])
        self.gamma_np = np.random.uniform(-0.05, 0.05, [self.cols])
        self.beta_np = np.random.uniform(-0.05, 0.05, [self.cols])
        self.epsilon = 1e-5
        self.place = paddle.CUDAPlace(0)
        self.residual_alpha = np.random.uniform(low=0.1, high=1.1, size=[1])

    def check_layernorm(self, x_np, gamma_np, beta_np, dtype):
        paddle.disable_static()
        x = paddle.to_tensor(x_np.astype(dtype))
        gamma = paddle.cast(
            paddle.to_tensor(gamma_np.astype(dtype)), paddle.float32
        )
        beta = paddle.cast(
            paddle.to_tensor(beta_np.astype(dtype)), paddle.float32
        )

        paddle_naive_norm_out = layernorm(x, gamma, beta, self.epsilon)
        paddle.enable_static()

        with paddle.static.program_guard(paddle.static.Program()):
            x_static = paddle.static.data(
                name="x_static", shape=[self.batch, self.cols], dtype=dtype
            )

            gamma_static = paddle.static.data(
                name="gamma_static", shape=[self.cols], dtype=paddle.float32
            )

            beta_static = paddle.static.data(
                name="beta_static", shape=[self.cols], dtype=paddle.float32
            )

            outs = paddle.incubate.nn.functional.norm_helper(
                x_static,
                None,
                None,
                gamma_static,
                beta_static,
                self.epsilon,
                1.0,
                norm_type="layernorm",
                begin_norm_axis=1,
            )

            exe = fluid.Executor(self.place)
            out_s = exe.run(
                feed={
                    "x_static": x_np.astype(dtype),
                    "gamma_static": gamma_np.astype(np.float32),
                    "beta_static": beta_np.astype(np.float32),
                },
                fetch_list=[outs],
            )

        return out_s, paddle_naive_norm_out

    def check_bias_add_residual_layernorm(
        self,
        x_np,
        residual_np,
        bias_np,
        gamma_np,
        beta_np,
        residual_alpha,
        dtype,
    ):
        paddle.disable_static()
        x = paddle.to_tensor(x_np.astype(dtype))
        residual = paddle.to_tensor(residual_np.astype(dtype))
        bias = paddle.to_tensor(bias_np.astype(dtype))
        gamma = paddle.cast(
            paddle.to_tensor(gamma_np.astype(dtype)), paddle.float32
        )
        beta = paddle.cast(
            paddle.to_tensor(beta_np.astype(dtype)), paddle.float32
        )

        paddle_naive_norm_out = residual_bias_add_layernorm(
            x, residual, bias, gamma, beta, self.epsilon, residual_alpha
        )

        paddle.enable_static()

        with paddle.static.program_guard(paddle.static.Program()):
            x_static = paddle.static.data(
                name="x_static", shape=[self.batch, self.cols], dtype=dtype
            )
            residual_static = paddle.static.data(
                name="residual_static",
                shape=[self.batch, self.cols],
                dtype=dtype,
            )
            bias_static = paddle.static.data(
                name="bias_static", shape=[self.cols], dtype=dtype
            )
            gamma_static = paddle.static.data(
                name="gamma_static", shape=[self.cols], dtype=paddle.float32
            )

            beta_static = paddle.static.data(
                name="beta_static", shape=[self.cols], dtype=paddle.float32
            )

            outs = paddle.incubate.nn.functional.norm_helper(
                x_static,
                residual_static,
                bias_static,
                gamma_static,
                beta_static,
                self.epsilon,
                residual_alpha,
                norm_type="layernorm",
                begin_norm_axis=1,
            )

            exe = fluid.Executor(self.place)
            out_s = exe.run(
                feed={
                    "x_static": x_np.astype(dtype),
                    "residual_static": residual_np.astype(dtype),
                    "bias_static": bias_np.astype(dtype),
                    "gamma_static": gamma_np.astype(np.float32),
                    "beta_static": beta_np.astype(np.float32),
                },
                fetch_list=[outs],
            )

        return out_s, paddle_naive_norm_out

    def check_rmsnorm(self, x_np, gamma_np, beta_np, dtype):
        paddle.disable_static()
        x = paddle.to_tensor(x_np.astype(dtype))
        gamma = paddle.to_tensor(gamma_np.astype(dtype))
        beta = paddle.to_tensor(beta_np.astype(dtype))

        paddle_naive_norm_out = rmsnorm(x, gamma, beta, self.epsilon)
        paddle.enable_static()

        with paddle.static.program_guard(paddle.static.Program()):
            x_static = paddle.static.data(
                name="x_static", shape=[self.batch, self.cols], dtype=dtype
            )

            gamma_static = paddle.static.data(
                name="gamma_static", shape=[self.cols], dtype=dtype
            )

            beta_static = paddle.static.data(
                name="beta_static", shape=[self.cols], dtype=dtype
            )

            outs = paddle.incubate.nn.functional.norm_helper(
                x_static,
                None,
                None,
                gamma_static,
                beta_static,
                self.epsilon,
                1.0,
                norm_type="rmsnorm",
                begin_norm_axis=1,
            )

            exe = fluid.Executor(self.place)
            out_s = exe.run(
                feed={
                    "x_static": x_np.astype(dtype),
                    "gamma_static": gamma_np.astype(dtype),
                    "beta_static": beta_np.astype(dtype),
                },
                fetch_list=[outs],
            )

        return out_s, paddle_naive_norm_out

    def check_bias_add_residual_rmsnorm(
        self, x_np, residual_np, bias_np, gamma_np, beta_np, dtype
    ):
        paddle.disable_static()
        x = paddle.to_tensor(x_np.astype(dtype))
        residual = paddle.to_tensor(residual_np.astype(dtype))
        bias = paddle.to_tensor(bias_np.astype(dtype))
        gamma = paddle.to_tensor(gamma_np.astype(dtype))
        beta = paddle.to_tensor(beta_np.astype(dtype))

        paddle_naive_norm_out = residual_bias_add_rmsnorm(
            x, residual, bias, gamma, beta, self.epsilon
        )
        paddle.enable_static()

        with paddle.static.program_guard(paddle.static.Program()):
            x_static = paddle.static.data(
                name="x_static", shape=[self.batch, self.cols], dtype=dtype
            )
            residual_static = paddle.static.data(
                name="residual_static",
                shape=[self.batch, self.cols],
                dtype=dtype,
            )
            bias_static = paddle.static.data(
                name="bias_static", shape=[self.cols], dtype=dtype
            )
            gamma_static = paddle.static.data(
                name="gamma_static", shape=[self.cols], dtype=dtype
            )

            beta_static = paddle.static.data(
                name="beta_static", shape=[self.cols], dtype=dtype
            )

            outs = paddle.incubate.nn.functional.norm_helper(
                x_static,
                residual_static,
                bias_static,
                gamma_static,
                beta_static,
                self.epsilon,
                1.0,
                norm_type="rmsnorm",
                begin_norm_axis=1,
            )

            exe = fluid.Executor(self.place)
            out_s = exe.run(
                feed={
                    "x_static": x_np.astype(dtype),
                    "residual_static": residual_np.astype(dtype),
                    "bias_static": bias_np.astype(dtype),
                    "gamma_static": gamma_np.astype(dtype),
                    "beta_static": beta_np.astype(dtype),
                },
                fetch_list=[outs],
            )

        return out_s, paddle_naive_norm_out

    def test_norm_helper_fp16(self):
        if not paddle.is_compiled_with_cuda():
            return

        paddle_norm_helper_rmsnorm, paddle_naive_rmsnorm = self.check_rmsnorm(
            self.x_np, self.gamma_np, self.beta_np, 'float16'
        )

        np.testing.assert_allclose(
            paddle_norm_helper_rmsnorm[0],
            paddle_naive_rmsnorm.numpy(),
            rtol=1e-3,
            atol=1e-3,
        )

        (
            paddle_norm_helper_bias_add_residual_rmsnorm,
            paddle_naive_bias_add_residual_rmsnorm,
        ) = self.check_bias_add_residual_rmsnorm(
            self.x_np,
            self.residual_np,
            self.bias_np,
            self.gamma_np,
            self.beta_np,
            'float16',
        )

        np.testing.assert_allclose(
            paddle_norm_helper_bias_add_residual_rmsnorm[0],
            paddle_naive_bias_add_residual_rmsnorm[0].numpy(),
            rtol=1e-3,
            atol=1e-3,
        )
        np.testing.assert_allclose(
            paddle_norm_helper_bias_add_residual_rmsnorm[1],
            paddle_naive_bias_add_residual_rmsnorm[1].numpy(),
            rtol=1e-3,
            atol=1e-3,
        )

        (
            paddle_norm_helper_layernorm,
            paddle_naive_layernorm,
        ) = self.check_layernorm(
            self.x_np, self.gamma_np, self.beta_np, 'float16'
        )

        np.testing.assert_allclose(
            paddle_norm_helper_layernorm[0],
            paddle_naive_layernorm.numpy(),
            rtol=1e-3,
            atol=1e-3,
        )

        (
            paddle_norm_helper_bias_add_residual_layernorm,
            paddle_naive_bias_add_residual_layernorm,
        ) = self.check_bias_add_residual_layernorm(
            self.x_np,
            self.residual_np,
            self.bias_np,
            self.gamma_np,
            self.beta_np,
            self.residual_alpha[0],
            'float16',
        )

        np.testing.assert_allclose(
            paddle_norm_helper_bias_add_residual_layernorm[0],
            paddle_naive_bias_add_residual_layernorm[0].numpy(),
            rtol=1e-3,
            atol=1e-3,
        )

        np.testing.assert_allclose(
            paddle_norm_helper_bias_add_residual_layernorm[1],
            paddle_naive_bias_add_residual_layernorm[1].numpy(),
            rtol=1e-3,
            atol=1e-3,
        )


if __name__ == "__main__":
    unittest.main()
