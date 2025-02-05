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
from paddle.base import core


def quant_helper(
    x, quant_scale, quant_round_type, quant_max_bound, quant_min_bound
):
    quant_value = quant_max_bound * quant_scale * x
    if quant_round_type == 0:
        quant_value = paddle.to_tensor(np.rint(quant_value.numpy()))
    else:
        quant_value = paddle.round(quant_value)
    return paddle.cast(
        paddle.clip(quant_value, quant_min_bound, quant_max_bound),
        paddle.int8,
    )


def naive_residual_bias_add(x, residual, bias, residual_alpha):
    return x + residual_alpha * residual + bias


def naive_layer_norm(x, gamma, beta, epsilon):
    x_float = paddle.cast(x, dtype=paddle.float32)
    mean = paddle.mean(x_float, axis=-1, keepdim=True)
    var = paddle.var(x_float, axis=-1, keepdim=True)
    sqrt_var = paddle.rsqrt(var + epsilon)
    normalized_output = (x_float - mean) * sqrt_var
    out = normalized_output * gamma + beta
    out = paddle.cast(out, x.dtype)
    return out


def naive_layer_norm_int8(
    x,
    gamma,
    beta,
    epsilon,
    in_scale,
    quant_round_type,
    quant_max_bound,
    quant_min_bound,
):
    out = naive_layer_norm(x, gamma, beta, epsilon)
    out = quant_helper(
        out, in_scale, quant_round_type, quant_max_bound, quant_min_bound
    )
    return out


def naive_residual_biasadd_layer_norm(
    x, residual, bias, gamma, beta, epsilon, residual_alpha
):
    x = x + residual * residual_alpha + bias
    residual_out = x
    mean = paddle.mean(x, axis=-1, keepdim=True)
    var = paddle.var(x, axis=-1, keepdim=True)
    sqrt_var = paddle.rsqrt(var + epsilon)
    out = ((x - mean) * sqrt_var) * paddle.cast(gamma, x.dtype) + paddle.cast(
        beta, x.dtype
    )
    return out, residual_out


def naive_residual_biasadd_layer_norm_int8(
    x,
    residual,
    bias,
    gamma,
    beta,
    epsilon,
    residual_alpha,
    in_scale,
    quant_round_type,
    quant_max_bound,
    quant_min_bound,
):
    out, residual_out = naive_residual_biasadd_layer_norm(
        x, residual, bias, gamma, beta, epsilon, residual_alpha
    )
    out = quant_helper(
        out, in_scale, quant_round_type, quant_max_bound, quant_min_bound
    )
    return out, residual_out


@unittest.skipIf(
    not core.is_compiled_with_cuda() and not paddle.is_compiled_with_rocm(),
    "core is not compiled with CUDA or ROCM",
)
class TestlayernormOp(unittest.TestCase):
    def setUp(self):
        np.random.seed(20)
        batch = 16
        cols = 256

        self.x_np = np.random.uniform(-0.05, 0.05, [batch, cols])
        self.residual_np = np.random.uniform(-0.05, 0.05, [batch, cols])
        self.bias_np = np.random.uniform(-0.05, 0.05, [cols])
        self.norm_weight_np = np.random.uniform(-0.05, 0.05, [cols])
        self.norm_bias_np = np.random.uniform(-0.05, 0.05, [cols])
        self.epsilon = 1e-5
        self.residual_alpha = np.random.uniform(low=0.1, high=1.1, size=[1])

        self.quant_scale = 0.15
        self.quant_round_type = 1
        self.quant_max_bound = 127
        self.quant_min_bound = -127

    def check_layernorm(self, x_np, gamma_np, beta_np, dtype):
        paddle.disable_static()
        x = paddle.to_tensor(x_np.astype(dtype))
        gamma = paddle.to_tensor(gamma_np.astype(np.float32))
        beta = paddle.to_tensor(beta_np.astype(np.float32))

        paddle_layernorm_out = paddle.incubate.nn.functional.fused_layer_norm(
            x, gamma, beta, self.epsilon, begin_norm_axis=1
        )
        paddle_naive_layernorm_out = naive_layer_norm(
            x, gamma, beta, self.epsilon
        )
        paddle.enable_static()
        return paddle_layernorm_out, paddle_naive_layernorm_out

    def check_layernorm_int8(self, x_np, gamma_np, beta_np, dtype):
        paddle.disable_static()
        x = paddle.to_tensor(x_np.astype(dtype))
        gamma = paddle.to_tensor(gamma_np.astype(np.float32))
        beta = paddle.to_tensor(beta_np.astype(np.float32))

        paddle_layernorm_out = paddle.incubate.nn.functional.fused_layer_norm(
            x,
            gamma,
            beta,
            self.epsilon,
            begin_norm_axis=1,
            quant_scale=self.quant_scale,
            quant_round_type=self.quant_round_type,
            quant_max_bound=self.quant_max_bound,
            quant_min_bound=self.quant_min_bound,
        )

        paddle_naive_layernorm_out = naive_layer_norm_int8(
            x,
            gamma,
            beta,
            self.epsilon,
            self.quant_scale,
            self.quant_round_type,
            self.quant_max_bound,
            self.quant_min_bound,
        )
        paddle.enable_static()
        return paddle_layernorm_out, paddle_naive_layernorm_out

    def check_residual_bias_add(self, x_np, residual_np, bias_np, dtype):
        paddle.disable_static()
        x = paddle.to_tensor(x_np.astype(dtype))
        residual = paddle.to_tensor(residual_np.astype(dtype))
        bias = paddle.to_tensor(bias_np.astype(dtype))

        paddle_layernorm_out = paddle.incubate.nn.functional.fused_layer_norm(
            x,
            None,
            None,
            self.epsilon,
            begin_norm_axis=1,
            bias=bias,
            residual=residual,
            residual_alpha=self.residual_alpha,
        )

        paddle_naive_residual_out = naive_residual_bias_add(
            x, residual, bias, self.residual_alpha
        )
        paddle.enable_static()
        return (paddle_layernorm_out, paddle_naive_residual_out)

    def check_residual_bias_layernorm(
        self, x_np, gamma_np, beta_np, residual_np, bias_np, dtype
    ):
        paddle.disable_static()
        x = paddle.to_tensor(x_np.astype(dtype))
        gamma = paddle.to_tensor(gamma_np.astype(np.float32))
        beta = paddle.to_tensor(beta_np.astype(np.float32))
        residual = paddle.to_tensor(residual_np.astype(dtype))
        bias = paddle.to_tensor(bias_np.astype(dtype))

        paddle_layernorm_out = paddle.incubate.nn.functional.fused_layer_norm(
            x,
            gamma,
            beta,
            self.epsilon,
            begin_norm_axis=1,
            bias=bias,
            residual=residual,
            residual_alpha=self.residual_alpha,
        )

        (
            paddle_naive_layernorm_out,
            paddle_naive_residual_out,
        ) = naive_residual_biasadd_layer_norm(
            x, residual, bias, gamma, beta, self.epsilon, self.residual_alpha
        )
        paddle.enable_static()
        return (
            paddle_layernorm_out,
            paddle_naive_layernorm_out,
            paddle_naive_residual_out,
        )

    def check_residual_bias_layernorm_int8(
        self, x_np, gamma_np, beta_np, residual_np, bias_np, dtype
    ):
        paddle.disable_static()
        x = paddle.to_tensor(x_np.astype(dtype))
        gamma = paddle.to_tensor(gamma_np.astype(np.float32))
        beta = paddle.to_tensor(beta_np.astype(np.float32))
        residual = paddle.to_tensor(residual_np.astype(dtype))
        bias = paddle.to_tensor(bias_np.astype(dtype))

        paddle_layernorm_out = paddle.incubate.nn.functional.fused_layer_norm(
            x,
            gamma,
            beta,
            self.epsilon,
            begin_norm_axis=1,
            bias=bias,
            residual=residual,
            residual_alpha=self.residual_alpha,
            quant_scale=self.quant_scale,
            quant_round_type=self.quant_round_type,
            quant_max_bound=self.quant_max_bound,
            quant_min_bound=self.quant_min_bound,
        )

        (
            paddle_naive_layernorm_out,
            paddle_naive_residual_out,
        ) = naive_residual_biasadd_layer_norm_int8(
            x,
            residual,
            bias,
            gamma,
            beta,
            self.epsilon,
            self.residual_alpha,
            self.quant_scale,
            self.quant_round_type,
            self.quant_max_bound,
            self.quant_min_bound,
        )
        paddle.enable_static()
        return (
            paddle_layernorm_out,
            paddle_naive_layernorm_out,
            paddle_naive_residual_out,
        )

    def test_residual_bias_add(self):
        if (
            not paddle.is_compiled_with_cuda()
            and not paddle.is_compiled_with_rocm()
        ):
            return
        (
            paddle_residual_bias_out,
            paddle_naive_residual_bias_out,
        ) = self.check_residual_bias_add(
            self.x_np, self.residual_np, self.bias_np, 'float16'
        )

        np.testing.assert_allclose(
            paddle_residual_bias_out[0].numpy(),
            paddle_naive_residual_bias_out.numpy(),
            rtol=1e-3,
            atol=1e-3,
        )

    def test_layernorm_fp16(self):
        if (
            not paddle.is_compiled_with_cuda()
            and not paddle.is_compiled_with_rocm()
        ):
            return
        paddle_layernorm, paddle_naive_layernorm = self.check_layernorm(
            self.x_np, self.norm_weight_np, self.norm_bias_np, 'float16'
        )

        np.testing.assert_allclose(
            paddle_layernorm[0].numpy(),
            paddle_naive_layernorm.numpy(),
            rtol=1e-3,
            atol=1e-3,
        )

    def test_layernorm_int8(self):
        if (
            not paddle.is_compiled_with_cuda()
            and not paddle.is_compiled_with_rocm()
        ):
            return
        paddle_layernorm, paddle_naive_layernorm = self.check_layernorm_int8(
            self.x_np, self.norm_weight_np, self.norm_bias_np, 'float16'
        )
        np.testing.assert_allclose(
            paddle_layernorm[0].numpy(),
            paddle_naive_layernorm.numpy(),
            rtol=2,
            atol=2,
        )

    def test_residual_bias_add_layernorm_fp16(self):
        if (
            not paddle.is_compiled_with_cuda()
            and not paddle.is_compiled_with_rocm()
        ):
            return
        (
            paddle_layernorm,
            paddle_naive_layernorm,
            paddle_naive_residual_out,
        ) = self.check_residual_bias_layernorm(
            self.x_np,
            self.norm_weight_np,
            self.norm_bias_np,
            self.residual_np,
            self.bias_np,
            'float16',
        )

        np.testing.assert_allclose(
            paddle_layernorm[0].numpy(),
            paddle_naive_layernorm.numpy(),
            rtol=1e-3,
            atol=1e-3,
        )

        np.testing.assert_allclose(
            paddle_layernorm[1].numpy(),
            paddle_naive_residual_out.numpy(),
            rtol=1e-3,
            atol=1e-3,
        )

    def test_residual_bias_add_layernorm_int8(self):
        if (
            not paddle.is_compiled_with_cuda()
            and not paddle.is_compiled_with_rocm()
        ):
            return
        (
            paddle_layernorm,
            paddle_naive_layernorm,
            paddle_naive_residual_out,
        ) = self.check_residual_bias_layernorm_int8(
            self.x_np,
            self.norm_weight_np,
            self.norm_bias_np,
            self.residual_np,
            self.bias_np,
            'float16',
        )

        np.testing.assert_allclose(
            paddle_layernorm[0].numpy(),
            paddle_naive_layernorm.numpy(),
            rtol=2,
            atol=2,
        )

        np.testing.assert_allclose(
            paddle_layernorm[1].numpy(),
            paddle_naive_residual_out.numpy(),
            rtol=1e-3,
            atol=1e-3,
        )


@unittest.skipIf(
    not core.is_compiled_with_cuda() and not paddle.is_compiled_with_rocm(),
    "core is not compiled with CUDA or ROCM",
)
class TestlayernormStaticOp(unittest.TestCase):
    def setUp(self):
        np.random.seed(20)
        self.batch = 16
        self.cols = 256

        self.x_np = np.random.uniform(-0.05, 0.05, [self.batch, self.cols])
        self.residual_np = np.random.uniform(
            -0.05, 0.05, [self.batch, self.cols]
        )
        self.bias_np = np.random.uniform(-0.05, 0.05, [self.cols])
        self.norm_weight_np = np.random.uniform(-0.05, 0.05, [self.cols])
        self.norm_bias_np = np.random.uniform(-0.05, 0.05, [self.cols])
        self.epsilon = 1e-5
        self.residual_alpha = np.random.uniform(low=0.1, high=1.1, size=[1])

        self.quant_scale = 0.15
        self.quant_round_type = 1
        self.quant_max_bound = 127
        self.quant_min_bound = -127
        self.place = paddle.CUDAPlace(0)

    def check_layernorm(self, x_np, gamma_np, beta_np, dtype):
        paddle.disable_static()
        x = paddle.to_tensor(x_np.astype(dtype))
        gamma = paddle.to_tensor(gamma_np.astype(np.float32))
        beta = paddle.to_tensor(beta_np.astype(np.float32))

        paddle_naive_layernorm_out = naive_layer_norm(
            x, gamma, beta, self.epsilon
        )
        paddle.enable_static()

        with paddle.static.program_guard(paddle.static.Program()):
            x_static = paddle.static.data(
                name="x_static", shape=[self.batch, self.cols], dtype=dtype
            )
            gamma_static = paddle.static.data(
                name="gamma_static", shape=[self.cols], dtype='float32'
            )
            beta_static = paddle.static.data(
                name="beta_static", shape=[self.cols], dtype='float32'
            )
            outs = paddle.incubate.nn.functional.fused_layer_norm(
                x_static,
                gamma_static,
                beta_static,
                self.epsilon,
                begin_norm_axis=1,
            )[0]
            exe = paddle.static.Executor(self.place)
            out_s = exe.run(
                feed={
                    "x_static": x_np.astype(dtype),
                    "gamma_static": gamma_np.astype(np.float32),
                    "beta_static": beta_np.astype(np.float32),
                },
                fetch_list=[outs],
            )
        return out_s, paddle_naive_layernorm_out

    def check_layernorm_int8(self, x_np, gamma_np, beta_np, dtype):
        paddle.disable_static()
        x = paddle.to_tensor(x_np.astype(dtype))
        gamma = paddle.to_tensor(gamma_np.astype(np.float32))
        beta = paddle.to_tensor(beta_np.astype(np.float32))

        paddle_naive_layernorm_out = naive_layer_norm_int8(
            x,
            gamma,
            beta,
            self.epsilon,
            self.quant_scale,
            self.quant_round_type,
            self.quant_max_bound,
            self.quant_min_bound,
        )
        paddle.enable_static()

        with paddle.static.program_guard(paddle.static.Program()):
            x_static = paddle.static.data(
                name="x_static", shape=[self.batch, self.cols], dtype=dtype
            )
            gamma_static = paddle.static.data(
                name="gamma_static", shape=[self.cols], dtype='float32'
            )
            beta_static = paddle.static.data(
                name="beta_static", shape=[self.cols], dtype='float32'
            )
            outs = paddle.incubate.nn.functional.fused_layer_norm(
                x_static,
                gamma_static,
                beta_static,
                self.epsilon,
                begin_norm_axis=1,
                quant_scale=self.quant_scale,
                quant_round_type=self.quant_round_type,
                quant_max_bound=self.quant_max_bound,
                quant_min_bound=self.quant_min_bound,
            )[0]
            exe = paddle.static.Executor(self.place)
            out_s = exe.run(
                feed={
                    "x_static": x_np.astype(dtype),
                    "gamma_static": gamma_np.astype(np.float32),
                    "beta_static": beta_np.astype(np.float32),
                },
                fetch_list=[outs],
            )
        return out_s, paddle_naive_layernorm_out

    def check_residual_bias_add(self, x_np, residual_np, bias_np, dtype):
        paddle.disable_static()
        x = paddle.to_tensor(x_np.astype(dtype))
        residual = paddle.to_tensor(residual_np.astype(dtype))
        bias = paddle.to_tensor(bias_np.astype(dtype))

        paddle_naive_residual_out = naive_residual_bias_add(
            x, residual, bias, self.residual_alpha
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
            outs = paddle.incubate.nn.functional.fused_layer_norm(
                x_static,
                None,
                None,
                self.epsilon,
                begin_norm_axis=1,
                bias=bias_static,
                residual=residual_static,
                residual_alpha=self.residual_alpha,
                quant_scale=self.quant_scale,
                quant_round_type=self.quant_round_type,
                quant_max_bound=self.quant_max_bound,
                quant_min_bound=self.quant_min_bound,
            )[0]

            exe = paddle.static.Executor(self.place)
            out_s = exe.run(
                feed={
                    "x_static": x_np.astype(dtype),
                    "residual_static": residual_np.astype(dtype),
                    "bias_static": bias_np.astype(dtype),
                },
                fetch_list=[
                    outs
                ],  # NOTE: Only fetch `out`, because `residual_out` will not be initialized if both `norm_weight` and `norm_bias` are None.
            )
        return out_s, paddle_naive_residual_out

    def check_residual_bias_layernorm(
        self, x_np, gamma_np, beta_np, residual_np, bias_np, dtype
    ):
        paddle.disable_static()
        x = paddle.to_tensor(x_np.astype(dtype))
        gamma = paddle.to_tensor(gamma_np.astype(np.float32))
        beta = paddle.to_tensor(beta_np.astype(np.float32))
        residual = paddle.to_tensor(residual_np.astype(dtype))
        bias = paddle.to_tensor(bias_np.astype(dtype))

        (
            paddle_naive_layernorm_out,
            paddle_naive_residual_out,
        ) = naive_residual_biasadd_layer_norm(
            x, residual, bias, gamma, beta, self.epsilon, self.residual_alpha
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
                name="gamma_static", shape=[self.cols], dtype='float32'
            )
            beta_static = paddle.static.data(
                name="beta_static", shape=[self.cols], dtype='float32'
            )
            outs, residual = paddle.incubate.nn.functional.fused_layer_norm(
                x_static,
                gamma_static,
                beta_static,
                self.epsilon,
                begin_norm_axis=1,
                residual_alpha=self.residual_alpha,
                bias=bias_static,
                residual=residual_static,
            )[:2]

            exe = paddle.static.Executor(self.place)
            out_s = exe.run(
                feed={
                    "x_static": x_np.astype(dtype),
                    "gamma_static": gamma_np.astype(np.float32),
                    "beta_static": beta_np.astype(np.float32),
                    "residual_static": residual_np.astype(dtype),
                    "bias_static": bias_np.astype(dtype),
                },
                fetch_list=[outs, residual],
            )
        return out_s, paddle_naive_layernorm_out, paddle_naive_residual_out

    def check_residual_bias_layernorm_int8(
        self, x_np, gamma_np, beta_np, residual_np, bias_np, dtype
    ):
        paddle.disable_static()
        x = paddle.to_tensor(x_np.astype(dtype))
        gamma = paddle.to_tensor(gamma_np.astype(np.float32))
        beta = paddle.to_tensor(beta_np.astype(np.float32))
        residual = paddle.to_tensor(residual_np.astype(dtype))
        bias = paddle.to_tensor(bias_np.astype(dtype))

        (
            paddle_naive_layernorm_out,
            paddle_naive_residual_out,
        ) = naive_residual_biasadd_layer_norm_int8(
            x,
            residual,
            bias,
            gamma,
            beta,
            self.epsilon,
            self.residual_alpha,
            self.quant_scale,
            self.quant_round_type,
            self.quant_max_bound,
            self.quant_min_bound,
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
                name="gamma_static", shape=[self.cols], dtype='float32'
            )
            beta_static = paddle.static.data(
                name="beta_static", shape=[self.cols], dtype='float32'
            )
            outs, residual = paddle.incubate.nn.functional.fused_layer_norm(
                x_static,
                gamma_static,
                beta_static,
                self.epsilon,
                begin_norm_axis=1,
                bias=bias_static,
                residual=residual_static,
                residual_alpha=self.residual_alpha,
                quant_scale=self.quant_scale,
                quant_round_type=self.quant_round_type,
                quant_max_bound=self.quant_max_bound,
                quant_min_bound=self.quant_min_bound,
            )[:2]

            exe = paddle.static.Executor(self.place)
            out_s = exe.run(
                feed={
                    "x_static": x_np.astype(dtype),
                    "gamma_static": gamma_np.astype(np.float32),
                    "beta_static": beta_np.astype(np.float32),
                    "residual_static": residual_np.astype(dtype),
                    "bias_static": bias_np.astype(dtype),
                },
                fetch_list=[outs, residual],
            )
        return out_s, paddle_naive_layernorm_out, paddle_naive_residual_out

    def test_layernorm_fp16(self):
        if (
            not paddle.is_compiled_with_cuda()
            and not paddle.is_compiled_with_rocm()
        ):
            return
        paddle_layernorm, paddle_naive_layernorm = self.check_layernorm(
            self.x_np, self.norm_weight_np, self.norm_bias_np, 'float16'
        )

        np.testing.assert_allclose(
            paddle_layernorm[0],
            paddle_naive_layernorm.numpy(),
            rtol=1e-3,
            atol=1e-3,
        )

    def test_layernorm_int8(self):
        if (
            not paddle.is_compiled_with_cuda()
            and not paddle.is_compiled_with_rocm()
        ):
            return
        paddle_layernorm, paddle_naive_layernorm = self.check_layernorm_int8(
            self.x_np, self.norm_weight_np, self.norm_bias_np, 'float16'
        )
        np.testing.assert_allclose(
            paddle_layernorm[0],
            paddle_naive_layernorm.numpy(),
            rtol=2,
            atol=2,
        )

    def test_residual_bias_add(self):
        if (
            not paddle.is_compiled_with_cuda()
            and not paddle.is_compiled_with_rocm()
        ):
            return
        (
            paddle_layernorm,
            paddle_naive_residual_out,
        ) = self.check_residual_bias_add(
            self.x_np,
            self.residual_np,
            self.bias_np,
            'float16',
        )

        np.testing.assert_allclose(
            paddle_layernorm[0],
            paddle_naive_residual_out.numpy(),
            rtol=1e-3,
            atol=1e-3,
        )

    def test_residual_bias_add_layernorm_fp16(self):
        if (
            not paddle.is_compiled_with_cuda()
            and not paddle.is_compiled_with_rocm()
        ):
            return
        (
            paddle_layernorm,
            paddle_naive_layernorm,
            paddle_naive_residual_out,
        ) = self.check_residual_bias_layernorm(
            self.x_np,
            self.norm_weight_np,
            self.norm_bias_np,
            self.residual_np,
            self.bias_np,
            'float16',
        )

        np.testing.assert_allclose(
            paddle_layernorm[0],
            paddle_naive_layernorm.numpy(),
            rtol=1e-3,
            atol=1e-3,
        )

        np.testing.assert_allclose(
            paddle_layernorm[1],
            paddle_naive_residual_out.numpy(),
            rtol=1e-3,
            atol=1e-3,
        )

    def test_residual_bias_add_layernorm_int8(self):
        if (
            not paddle.is_compiled_with_cuda()
            and not paddle.is_compiled_with_rocm()
        ):
            return
        (
            paddle_layernorm,
            paddle_naive_layernorm,
            paddle_naive_residual_out,
        ) = self.check_residual_bias_layernorm_int8(
            self.x_np,
            self.norm_weight_np,
            self.norm_bias_np,
            self.residual_np,
            self.bias_np,
            'float16',
        )

        np.testing.assert_allclose(
            paddle_layernorm[0],
            paddle_naive_layernorm.numpy(),
            rtol=2,
            atol=2,
        )

        np.testing.assert_allclose(
            paddle_layernorm[1],
            paddle_naive_residual_out.numpy(),
            rtol=2,
            atol=2,
        )


@unittest.skipIf(
    not core.supports_avx512f() or not core.is_compiled_with_avx(),
    "machine is not support AVX or is not compiled with AVX",
)
class TestlayernormOpCPU(unittest.TestCase):
    def setUp(self):
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        np.random.seed(20)
        batch = 16
        cols = 256

        self.x_np = np.random.uniform(-0.05, 0.05, [batch, cols])
        self.residual_np = np.random.uniform(-0.05, 0.05, [batch, cols])
        self.bias_np = np.random.uniform(-0.05, 0.05, [cols])
        self.norm_weight_np = np.random.uniform(-0.05, 0.05, [cols])
        self.norm_bias_np = np.random.uniform(-0.05, 0.05, [cols])
        self.epsilon = 1e-5
        self.residual_alpha = np.random.uniform(low=0.1, high=1.1, size=[1])

    def check_layernorm(self, x_np, gamma_np, beta_np, dtype):
        paddle.disable_static()
        x = paddle.to_tensor(x_np.astype(dtype))
        gamma = paddle.to_tensor(gamma_np.astype(np.float32))
        beta = paddle.to_tensor(beta_np.astype(np.float32))

        paddle_layernorm_out = paddle.incubate.nn.functional.fused_layer_norm(
            x, gamma, beta, self.epsilon, begin_norm_axis=1
        )[0]
        paddle_naive_layernorm_out = naive_layer_norm(
            x, gamma, beta, self.epsilon
        )
        paddle.enable_static()
        return paddle_layernorm_out, paddle_naive_layernorm_out

    def check_residual_bias_add(self, x_np, residual_np, bias_np, dtype):
        paddle.disable_static()
        x = paddle.to_tensor(x_np.astype(dtype))
        residual = paddle.to_tensor(residual_np.astype(dtype))
        bias = paddle.to_tensor(bias_np.astype(dtype))

        paddle_layernorm_out = paddle.incubate.nn.functional.fused_layer_norm(
            x,
            None,
            None,
            self.epsilon,
            begin_norm_axis=1,
            bias=bias,
            residual=residual,
            residual_alpha=self.residual_alpha,
        )[0]

        paddle_naive_residual_out = naive_residual_bias_add(
            x, residual, bias, self.residual_alpha
        )
        paddle.enable_static()
        return (paddle_layernorm_out, paddle_naive_residual_out)

    def check_residual_bias_layernorm(
        self, x_np, gamma_np, beta_np, residual_np, bias_np, dtype
    ):
        paddle.disable_static()
        x = paddle.to_tensor(x_np.astype(dtype))
        gamma = paddle.to_tensor(gamma_np.astype(np.float32))
        beta = paddle.to_tensor(beta_np.astype(np.float32))
        residual = paddle.to_tensor(residual_np.astype(dtype))
        bias = paddle.to_tensor(bias_np.astype(dtype))

        paddle_layernorm_out = paddle.incubate.nn.functional.fused_layer_norm(
            x,
            gamma,
            beta,
            self.epsilon,
            begin_norm_axis=1,
            bias=bias,
            residual=residual,
            residual_alpha=self.residual_alpha,
        )

        (
            paddle_naive_layernorm_out,
            paddle_naive_residual_out,
        ) = naive_residual_biasadd_layer_norm(
            x, residual, bias, gamma, beta, self.epsilon, self.residual_alpha
        )
        paddle.enable_static()
        return (
            paddle_layernorm_out,
            paddle_naive_layernorm_out,
            paddle_naive_residual_out,
        )

    def test_residual_bias_add(self):
        (
            paddle_residual_bias_out,
            paddle_naive_residual_bias_out,
        ) = self.check_residual_bias_add(
            self.x_np, self.residual_np, self.bias_np, 'float32'
        )
        np.testing.assert_allclose(
            paddle_residual_bias_out.numpy(),
            paddle_naive_residual_bias_out.numpy(),
            rtol=1e-3,
            atol=1e-3,
        )

    def test_layernorm(self):
        paddle_layernorm, paddle_naive_layernorm = self.check_layernorm(
            self.x_np, self.norm_weight_np, self.norm_bias_np, 'float32'
        )

        np.testing.assert_allclose(
            paddle_layernorm.numpy(),
            paddle_naive_layernorm.numpy(),
            rtol=1e-3,
            atol=1e-3,
        )

    def test_residual_bias_add_layernorm(self):
        (
            paddle_layernorm,
            paddle_naive_layernorm,
            paddle_naive_residual_out,
        ) = self.check_residual_bias_layernorm(
            self.x_np,
            self.norm_weight_np,
            self.norm_bias_np,
            self.residual_np,
            self.bias_np,
            'float32',
        )
        np.testing.assert_allclose(
            paddle_layernorm[0].numpy(),
            paddle_naive_layernorm.numpy(),
            rtol=1e-3,
            atol=1e-3,
        )
        np.testing.assert_allclose(
            paddle_layernorm[1].numpy(),
            paddle_naive_residual_out.numpy(),
            rtol=1e-3,
            atol=1e-3,
        )


@unittest.skipIf(
    not core.supports_avx512f() or not core.is_compiled_with_avx(),
    "machine is not support AVX or is not compiled with AVX",
)
class TestlayernormStaticOpCPU(unittest.TestCase):
    def setUp(self):
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        np.random.seed(20)
        self.batch = 16
        self.cols = 256

        self.x_np = np.random.uniform(-0.05, 0.05, [self.batch, self.cols])
        self.residual_np = np.random.uniform(
            -0.05, 0.05, [self.batch, self.cols]
        )
        self.bias_np = np.random.uniform(-0.05, 0.05, [self.cols])
        self.norm_weight_np = np.random.uniform(-0.05, 0.05, [self.cols])
        self.norm_bias_np = np.random.uniform(-0.05, 0.05, [self.cols])
        self.epsilon = 1e-5
        self.residual_alpha = np.random.uniform(low=0.1, high=1.1, size=[1])

        self.place = paddle.CPUPlace()

    def check_layernorm(self, x_np, gamma_np, beta_np, dtype):
        paddle.disable_static()
        x = paddle.to_tensor(x_np.astype(dtype))
        gamma = paddle.to_tensor(gamma_np.astype(np.float32))
        beta = paddle.to_tensor(beta_np.astype(np.float32))

        paddle_naive_layernorm_out = naive_layer_norm(
            x, gamma, beta, self.epsilon
        )
        paddle.enable_static()

        with paddle.static.program_guard(paddle.static.Program()):
            x_static = paddle.static.data(
                name="x_static", shape=[self.batch, self.cols], dtype=dtype
            )
            gamma_static = paddle.static.data(
                name="gamma_static", shape=[self.cols], dtype='float32'
            )
            beta_static = paddle.static.data(
                name="beta_static", shape=[self.cols], dtype='float32'
            )
            outs = paddle.incubate.nn.functional.fused_layer_norm(
                x_static,
                gamma_static,
                beta_static,
                self.epsilon,
                begin_norm_axis=1,
            )[0]
            exe = paddle.static.Executor(self.place)
            out_s = exe.run(
                feed={
                    "x_static": x_np.astype(dtype),
                    "gamma_static": gamma_np.astype(np.float32),
                    "beta_static": beta_np.astype(np.float32),
                },
                fetch_list=[outs],
            )
        return out_s, paddle_naive_layernorm_out

    def check_residual_bias_add(self, x_np, residual_np, bias_np, dtype):
        paddle.disable_static()
        x = paddle.to_tensor(x_np.astype(dtype))
        residual = paddle.to_tensor(residual_np.astype(dtype))
        bias = paddle.to_tensor(bias_np.astype(dtype))

        paddle_naive_residual_out = naive_residual_bias_add(
            x, residual, bias, self.residual_alpha
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
            outs = paddle.incubate.nn.functional.fused_layer_norm(
                x_static,
                None,
                None,
                self.epsilon,
                begin_norm_axis=1,
                bias=bias_static,
                residual=residual_static,
                residual_alpha=self.residual_alpha,
            )[0]

            exe = paddle.static.Executor(self.place)
            out_s = exe.run(
                feed={
                    "x_static": x_np.astype(dtype),
                    "residual_static": residual_np.astype(dtype),
                    "bias_static": bias_np.astype(dtype),
                },
                fetch_list=[
                    outs
                ],  # NOTE: Only fetch `out`, because `residual_out` will not be initialized if both `norm_weight` and `norm_bias` are None.
            )
        return out_s, paddle_naive_residual_out

    def check_residual_bias_layernorm(
        self, x_np, gamma_np, beta_np, residual_np, bias_np, dtype
    ):
        paddle.disable_static()
        x = paddle.to_tensor(x_np.astype(dtype))
        gamma = paddle.to_tensor(gamma_np.astype(np.float32))
        beta = paddle.to_tensor(beta_np.astype(np.float32))
        residual = paddle.to_tensor(residual_np.astype(dtype))
        bias = paddle.to_tensor(bias_np.astype(dtype))

        (
            paddle_naive_layernorm_out,
            paddle_naive_residual_out,
        ) = naive_residual_biasadd_layer_norm(
            x, residual, bias, gamma, beta, self.epsilon, self.residual_alpha
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
                name="gamma_static", shape=[self.cols], dtype='float32'
            )
            beta_static = paddle.static.data(
                name="beta_static", shape=[self.cols], dtype='float32'
            )
            outs, residual = paddle.incubate.nn.functional.fused_layer_norm(
                x_static,
                gamma_static,
                beta_static,
                self.epsilon,
                begin_norm_axis=1,
                residual_alpha=self.residual_alpha,
                bias=bias_static,
                residual=residual_static,
            )[:2]

            exe = paddle.static.Executor(self.place)
            out_s = exe.run(
                feed={
                    "x_static": x_np.astype(dtype),
                    "gamma_static": gamma_np.astype(np.float32),
                    "beta_static": beta_np.astype(np.float32),
                    "residual_static": residual_np.astype(dtype),
                    "bias_static": bias_np.astype(dtype),
                },
                fetch_list=[outs, residual],
            )
        return out_s, paddle_naive_layernorm_out, paddle_naive_residual_out

    def test_layernorm(self):
        paddle_layernorm, paddle_naive_layernorm = self.check_layernorm(
            self.x_np, self.norm_weight_np, self.norm_bias_np, 'float32'
        )

        np.testing.assert_allclose(
            paddle_layernorm[0],
            paddle_naive_layernorm.numpy(),
            rtol=1e-3,
            atol=1e-3,
        )

    def test_residual_bias_add(self):
        if (
            not paddle.is_compiled_with_cuda()
            and not paddle.is_compiled_with_rocm()
        ):
            return
        (
            paddle_layernorm,
            paddle_naive_residual_out,
        ) = self.check_residual_bias_add(
            self.x_np,
            self.residual_np,
            self.bias_np,
            'float32',
        )

        np.testing.assert_allclose(
            paddle_layernorm[0],
            paddle_naive_residual_out.numpy(),
            rtol=1e-3,
            atol=1e-3,
        )

    def test_residual_bias_add_layernorm(self):
        if (
            not paddle.is_compiled_with_cuda()
            and not paddle.is_compiled_with_rocm()
        ):
            return
        (
            paddle_layernorm,
            paddle_naive_layernorm,
            paddle_naive_residual_out,
        ) = self.check_residual_bias_layernorm(
            self.x_np,
            self.norm_weight_np,
            self.norm_bias_np,
            self.residual_np,
            self.bias_np,
            'float32',
        )

        np.testing.assert_allclose(
            paddle_layernorm[0],
            paddle_naive_layernorm.numpy(),
            rtol=1e-3,
            atol=1e-3,
        )

        np.testing.assert_allclose(
            paddle_layernorm[1],
            paddle_naive_residual_out.numpy(),
            rtol=1e-3,
            atol=1e-3,
        )


if __name__ == "__main__":
    unittest.main()
