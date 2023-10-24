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
from paddle import base
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


def naive_rms_norm(x, gamma, beta, epsilon):
    variance = x.pow(2).mean(-1, keepdim=True)
    out = paddle.rsqrt(variance + epsilon) * x
    out = out * gamma + beta
    return out


def naive_rms_norm_int8(
    x,
    gamma,
    beta,
    epsilon,
    in_scale,
    quant_round_type,
    quant_max_bound,
    quant_min_bound,
):
    out = naive_rms_norm(x, gamma, beta, epsilon)
    out = quant_helper(
        out, in_scale, quant_round_type, quant_max_bound, quant_min_bound
    )
    return out


def naive_residual_biasadd_rms_norm(x, residual, bias, gamma, beta, epsilon):
    x = x + residual + bias
    variance = x.pow(2).mean(-1, keepdim=True)
    out = paddle.rsqrt(variance + epsilon) * x
    out = out * gamma + beta
    return out


def naive_residual_biasadd_rms_norm_int8(
    x,
    residual,
    bias,
    gamma,
    beta,
    epsilon,
    in_scale,
    quant_round_type,
    quant_max_bound,
    quant_min_bound,
):
    out = naive_residual_biasadd_rms_norm(
        x, residual, bias, gamma, beta, epsilon
    )
    out = quant_helper(
        out, in_scale, quant_round_type, quant_max_bound, quant_min_bound
    )
    return out


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA "
)
class TestRMSNormOp(unittest.TestCase):
    def setUp(self):
        np.random.seed(20)
        batch = 32
        cols = 256
        self.x_np = np.random.random([batch, cols])
        self.residual_np = np.random.random([batch, cols])
        self.bias_np = np.random.random([cols])

        self.norm_weight_np = np.random.random([cols])
        self.norm_bias_np = np.random.random([cols])
        self.epsilon = 1e-6
        self.quant_scale = 0.15
        self.quant_round_type = 1
        self.quant_max_bound = 127
        self.quant_min_bound = -127

    def check_rmsnorm(self, x_np, gamma_np, beta_np, dtype):
        paddle.disable_static()
        x = paddle.to_tensor(x_np.astype(dtype))
        gamma = paddle.to_tensor(gamma_np.astype(dtype))
        beta = paddle.to_tensor(beta_np.astype(dtype))

        paddle_rmsnorm_out = paddle.incubate.nn.functional.fused_rms_norm(
            x, gamma, beta, self.epsilon, begin_norm_axis=1
        )
        paddle_naive_rmsnorm_out = naive_rms_norm(x, gamma, beta, self.epsilon)
        paddle.enable_static()
        return paddle_rmsnorm_out, paddle_naive_rmsnorm_out

    def check_rmsnorm_int8(self, x_np, gamma_np, beta_np, dtype):
        paddle.disable_static()
        x = paddle.to_tensor(x_np.astype(dtype))
        gamma = paddle.to_tensor(gamma_np.astype(dtype))
        beta = paddle.to_tensor(beta_np.astype(dtype))

        paddle_rmsnorm_out = paddle.incubate.nn.functional.fused_rms_norm(
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

        paddle_naive_rmsnorm_out = naive_rms_norm_int8(
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
        return paddle_rmsnorm_out, paddle_naive_rmsnorm_out

    def check_residual_bias_rmsnorm(
        self, x_np, gamma_np, beta_np, residual_np, bias_np, dtype
    ):
        paddle.disable_static()
        x = paddle.to_tensor(x_np.astype(dtype))
        gamma = paddle.to_tensor(gamma_np.astype(dtype))
        beta = paddle.to_tensor(beta_np.astype(dtype))
        residual = paddle.to_tensor(residual_np.astype(dtype))
        bias = paddle.to_tensor(bias_np.astype(dtype))

        paddle_rmsnorm_out = paddle.incubate.nn.functional.fused_rms_norm(
            x,
            gamma,
            beta,
            self.epsilon,
            begin_norm_axis=1,
            bias=bias,
            residual=residual,
        )

        paddle_naive_rmsnorm_out = naive_residual_biasadd_rms_norm(
            x, residual, bias, gamma, beta, self.epsilon
        )
        paddle.enable_static()
        return paddle_rmsnorm_out, paddle_naive_rmsnorm_out

    def check_residual_bias_rmsnorm_int8(
        self, x_np, gamma_np, beta_np, residual_np, bias_np, dtype
    ):
        paddle.disable_static()
        x = paddle.to_tensor(x_np.astype(dtype))
        gamma = paddle.to_tensor(gamma_np.astype(dtype))
        beta = paddle.to_tensor(beta_np.astype(dtype))
        residual = paddle.to_tensor(residual_np.astype(dtype))
        bias = paddle.to_tensor(bias_np.astype(dtype))

        paddle_rmsnorm_out = paddle.incubate.nn.functional.fused_rms_norm(
            x,
            gamma,
            beta,
            self.epsilon,
            begin_norm_axis=1,
            bias=bias,
            residual=residual,
            quant_scale=self.quant_scale,
            quant_round_type=self.quant_round_type,
            quant_max_bound=self.quant_max_bound,
            quant_min_bound=self.quant_min_bound,
        )

        paddle_naive_rmsnorm_out = naive_residual_biasadd_rms_norm_int8(
            x,
            residual,
            bias,
            gamma,
            beta,
            self.epsilon,
            self.quant_scale,
            self.quant_round_type,
            self.quant_max_bound,
            self.quant_min_bound,
        )
        paddle.enable_static()
        return paddle_rmsnorm_out, paddle_naive_rmsnorm_out

    def test_rmsnorm_fp16(self):
        if not paddle.is_compiled_with_cuda():
            return
        paddle_rmsnorm, paddle_naive_rmsnorm = self.check_rmsnorm(
            self.x_np, self.norm_weight_np, self.norm_bias_np, 'float16'
        )

        np.testing.assert_allclose(
            paddle_rmsnorm[0].numpy(),
            paddle_naive_rmsnorm.numpy(),
            rtol=1e-3,
            atol=1e-3,
        )

    def test_rmsnorm_int8(self):
        if not paddle.is_compiled_with_cuda():
            return
        paddle_rmsnorm, paddle_naive_rmsnorm = self.check_rmsnorm_int8(
            self.x_np, self.norm_weight_np, self.norm_bias_np, 'float16'
        )
        np.testing.assert_allclose(
            paddle_rmsnorm[0].numpy(),
            paddle_naive_rmsnorm.numpy(),
            rtol=2,
            atol=2,
        )

    def test_residual_bias_add_rmsnorm_fp16(self):
        if not paddle.is_compiled_with_cuda():
            return
        paddle_rmsnorm, paddle_naive_rmsnorm = self.check_residual_bias_rmsnorm(
            self.x_np,
            self.norm_weight_np,
            self.norm_bias_np,
            self.residual_np,
            self.bias_np,
            'float16',
        )

        np.testing.assert_allclose(
            paddle_rmsnorm[0].numpy(),
            paddle_naive_rmsnorm.numpy(),
            rtol=1e-3,
            atol=1e-3,
        )

    def test_residual_bias_add_rmsnorm_int8(self):
        if not paddle.is_compiled_with_cuda():
            return
        (
            paddle_rmsnorm,
            paddle_naive_rmsnorm,
        ) = self.check_residual_bias_rmsnorm_int8(
            self.x_np,
            self.norm_weight_np,
            self.norm_bias_np,
            self.residual_np,
            self.bias_np,
            'float16',
        )

        np.testing.assert_allclose(
            paddle_rmsnorm[0].numpy(),
            paddle_naive_rmsnorm.numpy(),
            rtol=2,
            atol=2,
        )


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA "
)
class TestRMSNormStaticOp(unittest.TestCase):
    def setUp(self):
        np.random.seed(20)
        self.batch = 32
        self.cols = 256
        self.x_np = np.random.random([self.batch, 256])
        self.norm_weight_np = np.random.random([256])
        self.norm_bias_np = np.random.random([256])
        self.residual_np = np.random.random([self.batch, 256])
        self.bias_np = np.random.random([256])
        self.epsilon = 1e-6
        self.quant_scale = 0.15
        self.quant_round_type = 1
        self.quant_max_bound = 127
        self.quant_min_bound = -127
        self.place = paddle.CUDAPlace(0)

    def check_rmsnorm(self, x_np, gamma_np, beta_np, dtype):
        paddle.disable_static()
        x = paddle.to_tensor(x_np.astype(dtype))
        gamma = paddle.to_tensor(gamma_np.astype(dtype))
        beta = paddle.to_tensor(beta_np.astype(dtype))

        paddle_naive_rmsnorm_out = naive_rms_norm(x, gamma, beta, self.epsilon)
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
            outs = paddle.incubate.nn.functional.fused_rms_norm(
                x_static,
                gamma_static,
                beta_static,
                self.epsilon,
                begin_norm_axis=1,
            )
            exe = base.Executor(self.place)
            out_s = exe.run(
                feed={
                    "x_static": x_np.astype(dtype),
                    "gamma_static": gamma_np.astype(dtype),
                    "beta_static": beta_np.astype(dtype),
                },
                fetch_list=[outs],
            )
        return out_s[0], paddle_naive_rmsnorm_out

    def test_rmsnorm_pir(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.x_np.astype("float32"))
        gamma = paddle.to_tensor(self.norm_weight_np.astype("float32"))
        beta = paddle.to_tensor(self.norm_bias_np.astype("float32"))

        paddle_naive_rmsnorm_out = naive_rms_norm(x, gamma, beta, self.epsilon)
        paddle.enable_static()

        with paddle.pir_utils.IrGuard():
            x_static = paddle.static.data(
                name="x_static", shape=[self.batch, self.cols], dtype="float32"
            )
            gamma_static = paddle.static.data(
                name="gamma_static", shape=[self.cols], dtype="float32"
            )
            beta_static = paddle.static.data(
                name="beta_static", shape=[self.cols], dtype="float32"
            )
            out, _ = paddle.incubate.nn.functional.fused_rms_norm(
                x_static,
                gamma_static,
                beta_static,
                self.epsilon,
                begin_norm_axis=1,
            )
            exe = base.Executor(self.place)
            out_s = exe.run(
                feed={
                    "x_static": self.x_np.astype("float32"),
                    "gamma_static": self.norm_weight_np.astype("float32"),
                    "beta_static": self.norm_bias_np.astype("float32"),
                },
                fetch_list=[out],
            )

        np.testing.assert_allclose(
            out_s[0],
            paddle_naive_rmsnorm_out.numpy(),
            rtol=1e-3,
            atol=1e-3,
        )

    def check_rmsnorm_int8(self, x_np, gamma_np, beta_np, dtype):
        paddle.disable_static()
        x = paddle.to_tensor(x_np.astype(dtype))
        gamma = paddle.to_tensor(gamma_np.astype(dtype))
        beta = paddle.to_tensor(beta_np.astype(dtype))

        paddle_naive_rmsnorm_out = naive_rms_norm_int8(
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
                name="gamma_static", shape=[self.cols], dtype=dtype
            )
            beta_static = paddle.static.data(
                name="beta_static", shape=[self.cols], dtype=dtype
            )
            outs = paddle.incubate.nn.functional.fused_rms_norm(
                x_static,
                gamma_static,
                beta_static,
                self.epsilon,
                begin_norm_axis=1,
                quant_scale=self.quant_scale,
                quant_round_type=self.quant_round_type,
                quant_max_bound=self.quant_max_bound,
                quant_min_bound=self.quant_min_bound,
            )
            exe = base.Executor(self.place)
            out_s = exe.run(
                feed={
                    "x_static": x_np.astype(dtype),
                    "gamma_static": gamma_np.astype(dtype),
                    "beta_static": beta_np.astype(dtype),
                },
                fetch_list=[outs],
            )
        return out_s[0], paddle_naive_rmsnorm_out

    def check_residual_bias_rmsnorm(
        self, x_np, gamma_np, beta_np, residual_np, bias_np, dtype
    ):
        paddle.disable_static()
        x = paddle.to_tensor(x_np.astype(dtype))
        gamma = paddle.to_tensor(gamma_np.astype(dtype))
        beta = paddle.to_tensor(beta_np.astype(dtype))
        residual = paddle.to_tensor(residual_np.astype(dtype))
        bias = paddle.to_tensor(bias_np.astype(dtype))

        paddle_naive_rmsnorm_out = naive_residual_biasadd_rms_norm(
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
            outs = paddle.incubate.nn.functional.fused_rms_norm(
                x_static,
                gamma_static,
                beta_static,
                self.epsilon,
                begin_norm_axis=1,
                bias=bias_static,
                residual=residual_static,
            )

            exe = base.Executor(self.place)
            out_s = exe.run(
                feed={
                    "x_static": x_np.astype(dtype),
                    "gamma_static": gamma_np.astype(dtype),
                    "beta_static": beta_np.astype(dtype),
                    "residual_static": residual_np.astype(dtype),
                    "bias_static": bias_np.astype(dtype),
                },
                fetch_list=[outs],
            )
        return out_s[0], paddle_naive_rmsnorm_out

    def test_rmsnorm_fp16(self):
        if not paddle.is_compiled_with_cuda():
            return
        paddle_rmsnorm, paddle_naive_rmsnorm = self.check_rmsnorm(
            self.x_np, self.norm_weight_np, self.norm_bias_np, 'float16'
        )

        np.testing.assert_allclose(
            paddle_rmsnorm,
            paddle_naive_rmsnorm.numpy(),
            rtol=1e-3,
            atol=1e-3,
        )

    def test_residual_bias_add_rmsnorm_fp16(self):
        if not paddle.is_compiled_with_cuda():
            return
        paddle_rmsnorm, paddle_naive_rmsnorm = self.check_residual_bias_rmsnorm(
            self.x_np,
            self.norm_weight_np,
            self.norm_bias_np,
            self.residual_np,
            self.bias_np,
            'float16',
        )

        np.testing.assert_allclose(
            paddle_rmsnorm,
            paddle_naive_rmsnorm.numpy(),
            rtol=1e-3,
            atol=1e-3,
        )

    def test_rmsnorm_int8(self):
        if not paddle.is_compiled_with_cuda():
            return
        paddle_rmsnorm, paddle_naive_rmsnorm = self.check_rmsnorm_int8(
            self.x_np, self.norm_weight_np, self.norm_bias_np, 'float16'
        )
        np.testing.assert_allclose(
            paddle_rmsnorm,
            paddle_naive_rmsnorm.numpy(),
            rtol=2,
            atol=2,
        )


if __name__ == "__main__":
    unittest.main()
