# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from paddle import fluid
from paddle.fluid import core
from paddle.fluid.framework import in_dygraph_mode
from paddle.fluid.layer_helper import LayerHelper


def layer_norm_int8_wrapper(
    x,
    residual,
    bias,
    norm_weight,
    norm_bias,
    epsilon,
    in_scale=-1,
    quant_round_type=0,
    quant_max_bound=0,
    quant_min_bound=0,
):
    if in_dygraph_mode():
        return paddle._C_ops.layer_norm_int8(
            x,
            residual,
            bias,
            norm_weight,
            norm_bias,
            epsilon,
            in_scale,
            quant_round_type,
            quant_max_bound,
            quant_min_bound,
            1,
        )

    helper = LayerHelper('layer_norm_int8', **locals())
    residual_out = helper.create_variable_for_type_inference(dtype=x.dtype)
    out = helper.create_variable_for_type_inference(dtype=paddle.int8)

    helper.append_op(
        type='layer_norm_int8',
        inputs={
            'x': x,
            'residual': residual,
            'bias': bias,
            'norm_weight': norm_weight,
            'norm_bias': norm_bias,
        },
        attrs={
            'in_scale': in_scale,
            'epsilon': epsilon,
            'quant_round_type': quant_round_type,
            'quant_max_bound': quant_max_bound,
            'quant_min_bound': quant_min_bound,
            'begin_norm_axis': 1,
        },
        outputs={'residual_out': residual_out, 'out': out},
    )
    return residual_out, out


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestLayerNormInt8Op(unittest.TestCase):
    def setUp(self):
        # np.random.seed(20)
        np.random.seed(0)
        self.batch = 32
        self.cols = 256
        self.x_np = np.random.uniform(-0.05, 0.05, [self.batch, self.cols])
        self.residual_np = np.random.uniform(
            -0.05, 0.05, [self.batch, self.cols]
        )
        self.bias_np = np.random.uniform(-0.05, 0.05, [self.cols])
        self.gamma_np = np.random.uniform(-0.05, 0.05, [self.cols])
        self.beta_np = np.random.uniform(-0.05, 0.05, [self.cols])
        self.epsilon = 1e-5
        self.in_scale = 0.15
        self.quant_round_type = 1
        self.quant_max_bound = 127
        self.quant_min_bound = -127

    def quant_helper(
        self, x, quant_scale, quant_round_type, quant_max_bound, quant_min_bound
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

    def layer_norm_int8_naive(
        self,
        x,
        residual,
        bias,
        norm_weight,
        norm_bias,
        epsilon,
        in_scale,
        quant_round_type,
        quant_max_bound,
        quant_min_bound,
    ):
        skip_residual_bias_add = x + residual + bias
        residual_out = skip_residual_bias_add
        mean = paddle.mean(skip_residual_bias_add, axis=-1, keepdim=True)
        var = paddle.var(skip_residual_bias_add, axis=-1, keepdim=True)
        sqrt_var = paddle.rsqrt(var + epsilon)

        normalized_out = ((residual_out - mean) * sqrt_var) * paddle.cast(
            norm_weight, x.dtype
        ) + paddle.cast(norm_bias, x.dtype)
        normalized_out = self.quant_helper(
            normalized_out,
            in_scale,
            quant_round_type,
            quant_max_bound,
            quant_min_bound,
        )
        return residual_out, normalized_out

    def check_main(
        self, x_np, residual_np, bias_np, norm_gamma_np, norm_beta_np, dtype
    ):
        paddle.disable_static()
        x = paddle.to_tensor(x_np.astype(dtype))
        residual = paddle.to_tensor(residual_np.astype(dtype))
        bias = paddle.to_tensor(bias_np.astype(dtype))
        norm_gamma = paddle.to_tensor(norm_gamma_np.astype(np.float32))
        norm_beta = paddle.to_tensor(norm_beta_np.astype(np.float32))

        paddle_layernorm_out = layer_norm_int8_wrapper(
            x,
            residual,
            bias,
            norm_gamma,
            norm_beta,
            self.epsilon,
            self.in_scale,
            self.quant_round_type,
            self.quant_max_bound,
            self.quant_min_bound,
        )
        paddle_naive_layernorm_out = self.layer_norm_int8_naive(
            x,
            residual,
            bias,
            norm_gamma,
            norm_beta,
            self.epsilon,
            self.in_scale,
            self.quant_round_type,
            self.quant_max_bound,
            self.quant_min_bound,
        )
        paddle.enable_static()
        return paddle_layernorm_out, paddle_naive_layernorm_out

    def test_layernorm_fp16(self):
        if not paddle.is_compiled_with_cuda():
            return

        paddle_rmsnorm, paddle_naive_rmsnorm = self.check_main(
            self.x_np,
            self.residual_np,
            self.bias_np,
            self.gamma_np,
            self.beta_np,
            'float16',
        )

        np.testing.assert_allclose(
            paddle_rmsnorm[0].numpy(),
            paddle_naive_rmsnorm[0].numpy(),
            rtol=1e-2,
            atol=1e-2,
        )

        np.testing.assert_allclose(
            paddle_rmsnorm[1].numpy(),
            paddle_naive_rmsnorm[1].numpy(),
            rtol=2,
            atol=2,
        )


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestLayerNormStaticInt8Op(unittest.TestCase):
    def setUp(self):
        # np.random.seed(20)
        np.random.seed(0)
        self.batch = 32
        self.cols = 256
        self.x_np = np.random.uniform(-0.05, 0.05, [self.batch, self.cols])
        self.residual_np = np.random.uniform(
            -0.05, 0.05, [self.batch, self.cols]
        )
        self.bias_np = np.random.uniform(-0.05, 0.05, [self.cols])
        self.gamma_np = np.random.uniform(-0.05, 0.05, [self.cols])
        self.beta_np = np.random.uniform(-0.05, 0.05, [self.cols])
        self.epsilon = 1e-5
        self.in_scale = 0.15
        self.quant_round_type = 1
        self.quant_max_bound = 127
        self.quant_min_bound = -127
        self.place = paddle.CUDAPlace(0)

    def quant_helper(
        self, x, quant_scale, quant_round_type, quant_max_bound, quant_min_bound
    ):
        print(quant_max_bound, quant_scale)
        quant_value = quant_max_bound * quant_scale * x
        if quant_round_type == 0:
            quant_value = paddle.to_tensor(np.rint(quant_value.numpy()))
        else:
            quant_value = paddle.round(quant_value)
        return paddle.cast(
            paddle.clip(quant_value, quant_min_bound, quant_max_bound),
            paddle.int8,
        )

    def layer_norm_int8_naive(
        self,
        x,
        residual,
        bias,
        norm_weight,
        norm_bias,
        epsilon,
        in_scale,
        quant_round_type,
        quant_max_bound,
        quant_min_bound,
    ):
        skip_residual_bias_add = x + residual + bias
        residual_out = skip_residual_bias_add
        mean = paddle.mean(skip_residual_bias_add, axis=-1, keepdim=True)
        var = paddle.var(skip_residual_bias_add, axis=-1, keepdim=True)
        sqrt_var = paddle.rsqrt(var + epsilon)

        normalized_out = ((residual_out - mean) * sqrt_var) * paddle.cast(
            norm_weight, x.dtype
        ) + paddle.cast(norm_bias, x.dtype)
        normalized_out = self.quant_helper(
            normalized_out,
            in_scale,
            quant_round_type,
            quant_max_bound,
            quant_min_bound,
        )
        return residual_out, normalized_out

    def check_main(
        self, x_np, residual_np, bias_np, norm_gamma_np, norm_beta_np, dtype
    ):
        paddle.disable_static()
        x = paddle.to_tensor(x_np.astype(dtype))
        residual = paddle.to_tensor(residual_np.astype(dtype))
        bias = paddle.to_tensor(bias_np.astype(dtype))
        norm_gamma = paddle.to_tensor(norm_gamma_np.astype(np.float32))
        norm_beta = paddle.to_tensor(norm_beta_np.astype(np.float32))
        paddle_naive_layernorm_out = self.layer_norm_int8_naive(
            x,
            residual,
            bias,
            norm_gamma,
            norm_beta,
            self.epsilon,
            self.in_scale,
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
                name="gamma_static", shape=[self.cols], dtype=paddle.float32
            )

            beta_static = paddle.static.data(
                name="beta_static", shape=[self.cols], dtype=paddle.float32
            )

            outs = layer_norm_int8_wrapper(
                x_static,
                residual_static,
                bias_static,
                gamma_static,
                beta_static,
                self.epsilon,
                self.in_scale,
                self.quant_round_type,
                self.quant_max_bound,
                self.quant_min_bound,
            )

            exe = fluid.Executor(self.place)
            out_s = exe.run(
                feed={
                    "x_static": x_np.astype(dtype),
                    "residual_static": residual_np.astype(dtype),
                    "bias_static": bias_np.astype(dtype),
                    "gamma_static": norm_gamma_np.astype(np.float32),
                    "beta_static": norm_beta_np.astype(np.float32),
                },
                fetch_list=[outs],
            )

        return out_s, paddle_naive_layernorm_out

    def test_layernorm_fp16(self):
        if not paddle.is_compiled_with_cuda():
            return

        paddle_rmsnorm, paddle_naive_rmsnorm = self.check_main(
            self.x_np,
            self.residual_np,
            self.bias_np,
            self.gamma_np,
            self.beta_np,
            'float16',
        )

        np.testing.assert_allclose(
            paddle_rmsnorm[0],
            paddle_naive_rmsnorm[0].numpy(),
            rtol=1e-2,
            atol=1e-2,
        )

        np.testing.assert_allclose(
            paddle_rmsnorm[1],
            paddle_naive_rmsnorm[1].numpy(),
            rtol=2,
            atol=2,
        )


if __name__ == '__main__':
    unittest.main()
