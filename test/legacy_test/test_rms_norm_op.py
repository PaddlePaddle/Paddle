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


def naive_residual_bias_add(x, residual, bias):
    return x + residual + bias


def naive_rms_norm(x, gamma, beta=None, epsilon=1e-5):
    variance = x.pow(2).mean(-1, keepdim=True)
    out = paddle.rsqrt(variance + epsilon) * x
    out = out * gamma
    if beta is not None:
        out = out + beta
    return out


def fused_rms_norm(x, gamma, beta=None, epsilon=1e-5, begin_norm_axis=1):
    out = paddle.incubate.nn.functional.fused_rms_norm(
        x, gamma, beta, epsilon, begin_norm_axis=begin_norm_axis
    )
    return out[0]


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
    not core.is_compiled_with_cuda() and not paddle.is_compiled_with_rocm(),
    "core is not compiled with CUDA or ROCM",
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
        )[0]
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
        )[0]

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
        )[0]

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
        )[0]

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
        if (
            not paddle.is_compiled_with_cuda()
            and not paddle.is_compiled_with_rocm()
        ):
            return
        paddle_rmsnorm, paddle_naive_rmsnorm = self.check_rmsnorm(
            self.x_np, self.norm_weight_np, self.norm_bias_np, 'float16'
        )

        np.testing.assert_allclose(
            paddle_rmsnorm.numpy(),
            paddle_naive_rmsnorm.numpy(),
            rtol=1e-3,
            atol=1e-3,
        )

    def test_rmsnorm_int8(self):
        if (
            not paddle.is_compiled_with_cuda()
            and not paddle.is_compiled_with_rocm()
        ):
            return
        paddle_rmsnorm, paddle_naive_rmsnorm = self.check_rmsnorm_int8(
            self.x_np, self.norm_weight_np, self.norm_bias_np, 'float16'
        )
        np.testing.assert_allclose(
            paddle_rmsnorm.numpy(),
            paddle_naive_rmsnorm.numpy(),
            rtol=2,
            atol=2,
        )

    def test_residual_bias_add_rmsnorm_fp16(self):
        if (
            not paddle.is_compiled_with_cuda()
            and not paddle.is_compiled_with_rocm()
        ):
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
            paddle_rmsnorm.numpy(),
            paddle_naive_rmsnorm.numpy(),
            rtol=1e-3,
            atol=1e-3,
        )

    def test_residual_bias_add_rmsnorm_int8(self):
        if (
            not paddle.is_compiled_with_cuda()
            and not paddle.is_compiled_with_rocm()
        ):
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
            paddle_rmsnorm.numpy(),
            paddle_naive_rmsnorm.numpy(),
            rtol=2,
            atol=2,
        )

    def test_rms_norm_backward(self):
        def get_paddle_tensor(shape, dtype, bound=0.5):
            tmp = paddle.uniform(shape, dtype=dtype, min=-bound, max=bound)
            tmp.stop_gradient = False
            return tmp

        def get_forward_backward(func, seed, dtype):
            paddle.disable_static()
            paddle.seed(seed)
            x = get_paddle_tensor([2, 256], dtype)
            scale = get_paddle_tensor([256], dtype)
            out_g = paddle.randn([2, 256], dtype)
            out = func(x, scale)
            paddle.autograd.backward([out], [out_g], True)
            return out, (x.grad, scale.grad)

        dtypes = [paddle.float32]
        if paddle.amp.is_bfloat16_supported('gpu'):
            dtypes.append(paddle.bfloat16)
        if paddle.amp.is_float16_supported('gpu'):
            dtypes.append(paddle.float16)
        for dtype in dtypes:
            raw_out, raw_grads = get_forward_backward(
                naive_rms_norm, seed=2024, dtype=dtype
            )
            fused_out, fused_grads = get_forward_backward(
                fused_rms_norm, seed=2024, dtype=dtype
            )
            # forward rtol
            rtol = 1e-5 if dtype == paddle.float32 else 1e-2
            np.testing.assert_allclose(
                raw_out.astype(paddle.float32).numpy(),
                fused_out.astype(paddle.float32).numpy(),
                rtol=rtol,
            )
            # backward rtol, only check float32 grad
            rtol = 1e-3
            if dtype == paddle.float32:
                raw_x_grad, raw_scale_grad = raw_grads
                fused_x_grad, fused_scale_grad = fused_grads
                np.testing.assert_allclose(
                    raw_x_grad.astype(paddle.float32).numpy(),
                    fused_x_grad.astype(paddle.float32).numpy(),
                    rtol=rtol,
                )
                np.testing.assert_allclose(
                    raw_scale_grad.astype(paddle.float32).numpy(),
                    fused_scale_grad.astype(paddle.float32).numpy(),
                    rtol=rtol,
                )


@unittest.skipIf(
    not core.is_compiled_with_cuda() and not paddle.is_compiled_with_rocm(),
    "core is not compiled with CUDA or ROCM",
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
            )[0]
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
            )[0]
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
            )[0]

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
        if (
            not paddle.is_compiled_with_cuda()
            and not paddle.is_compiled_with_rocm()
        ):
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
        if (
            not paddle.is_compiled_with_cuda()
            and not paddle.is_compiled_with_rocm()
        ):
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
        if (
            not paddle.is_compiled_with_cuda()
            and not paddle.is_compiled_with_rocm()
        ):
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


@unittest.skipIf(
    not core.supports_avx512f() or not core.is_compiled_with_avx(),
    "machine is not support AVX or is not compiled with AVX",
)
class TestRMSNormOpCPU(unittest.TestCase):
    def setUp(self):
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        np.random.seed(20)
        batch = 32
        cols = 256
        self.x_np = np.random.random([batch, cols])
        self.residual_np = np.random.random([batch, cols])
        self.bias_np = np.random.random([cols])

        self.norm_weight_np = np.random.random([cols])
        self.norm_bias_np = np.random.random([cols])
        self.epsilon = 1e-6

    def check_rmsnorm(self, x_np, gamma_np, beta_np, dtype):
        paddle.disable_static()
        x = paddle.to_tensor(x_np.astype(dtype))
        gamma = paddle.to_tensor(gamma_np.astype(dtype))
        beta = paddle.to_tensor(beta_np.astype(dtype))

        paddle_rmsnorm_out = paddle.incubate.nn.functional.fused_rms_norm(
            x, gamma, beta, self.epsilon, begin_norm_axis=1
        )[0]
        paddle_naive_rmsnorm_out = naive_rms_norm(x, gamma, beta, self.epsilon)
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

        paddle_naive_residual_out = naive_residual_bias_add(x, residual, bias)
        paddle.enable_static()
        return (
            paddle_rmsnorm_out,
            paddle_naive_rmsnorm_out,
            paddle_naive_residual_out,
        )

    def test_rmsnorm(self):
        paddle_rmsnorm, paddle_naive_rmsnorm = self.check_rmsnorm(
            self.x_np, self.norm_weight_np, self.norm_bias_np, 'float32'
        )
        np.testing.assert_allclose(
            paddle_rmsnorm.numpy(),
            paddle_naive_rmsnorm.numpy(),
            rtol=1e-3,
            atol=1e-3,
        )

    def test_residual_bias_add_rmsnorm(self):
        (
            paddle_rmsnorm,
            paddle_naive_rmsnorm,
            paddle_naive_residual_out,
        ) = self.check_residual_bias_rmsnorm(
            self.x_np,
            self.norm_weight_np,
            self.norm_bias_np,
            self.residual_np,
            self.bias_np,
            'float32',
        )

        np.testing.assert_allclose(
            paddle_rmsnorm[0].numpy(),
            paddle_naive_rmsnorm.numpy(),
            rtol=1e-3,
            atol=1e-3,
        )
        np.testing.assert_allclose(
            paddle_rmsnorm[1].numpy(),
            paddle_naive_residual_out.numpy(),
            rtol=1e-3,
            atol=1e-3,
        )


@unittest.skipIf(
    not core.supports_avx512f() or not core.is_compiled_with_avx(),
    "machine is not support AVX or is not compiled with AVX",
)
class TestRMSNormStaticOpCPU(unittest.TestCase):
    def setUp(self):
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        np.random.seed(20)
        self.batch = 32
        self.cols = 256
        self.x_np = np.random.random([self.batch, 256])
        self.norm_weight_np = np.random.random([256])
        self.norm_bias_np = np.random.random([256])
        self.residual_np = np.random.random([self.batch, 256])
        self.bias_np = np.random.random([256])
        self.epsilon = 1e-6
        self.place = paddle.CPUPlace()

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
            )[0]
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
            )[0]

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

    def test_rmsnorm(self):
        if (
            not paddle.is_compiled_with_cuda()
            and not paddle.is_compiled_with_rocm()
        ):
            return
        paddle_rmsnorm, paddle_naive_rmsnorm = self.check_rmsnorm(
            self.x_np, self.norm_weight_np, self.norm_bias_np, 'float32'
        )

        np.testing.assert_allclose(
            paddle_rmsnorm,
            paddle_naive_rmsnorm.numpy(),
            rtol=1e-3,
            atol=1e-3,
        )

    def test_residual_bias_add_rmsnorm(self):
        if (
            not paddle.is_compiled_with_cuda()
            and not paddle.is_compiled_with_rocm()
        ):
            return
        paddle_rmsnorm, paddle_naive_rmsnorm = self.check_residual_bias_rmsnorm(
            self.x_np,
            self.norm_weight_np,
            self.norm_bias_np,
            self.residual_np,
            self.bias_np,
            'float32',
        )

        np.testing.assert_allclose(
            paddle_rmsnorm,
            paddle_naive_rmsnorm.numpy(),
            rtol=1e-3,
            atol=1e-3,
        )


if __name__ == "__main__":
    unittest.main()
