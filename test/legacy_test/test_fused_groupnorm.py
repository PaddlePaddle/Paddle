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
import paddle.nn.functional as F
from paddle import base
from paddle.base import core
from paddle.base.layer_helper import LayerHelper


def naive_residual_add(x, residual):
    return np.add(x, residual)


def naive_group_norm(x, scale, bias, epsilon, groups, data_layout):
    dim = x.ndim
    if dim == 3:
        if data_layout == "NHWC":
            x = np.transpose(x, (0, 2, 1))  # NLC => NCL
        N, C, L = x.shape
        G = groups
        x = x.reshape((N * G, -1))
        mean = np.mean(x, axis=1, keepdims=True)
        var = np.var(x, axis=1, keepdims=True)
        output = (x - mean) / np.sqrt(var + epsilon)
        output = output.reshape((N, C, L)) * scale.reshape(
            (-1, 1)
        ) + bias.reshape((-1, 1))
        if data_layout == "NHWC":
            output = np.transpose(output, (0, 2, 1))  # NCL => NLC
        return [output, mean.reshape((N, G)), var.reshape((N, G))]
    elif dim == 4:
        if data_layout == "NHWC":
            x = np.transpose(x, (0, 3, 1, 2))  # NHWC => NCHW
        N, C, H, W = x.shape
        G = groups
        x = x.reshape((N * G, -1))
        mean = np.mean(x, axis=1, keepdims=True)
        var = np.var(x, axis=1, keepdims=True)
        output = (x - mean) / np.sqrt(var + epsilon)
        output = output.reshape((N, C, H, W)) * scale.reshape(
            (-1, 1, 1)
        ) + bias.reshape((-1, 1, 1))
        if data_layout == "NHWC":
            output = np.transpose(output, (0, 2, 3, 1))  # NCHW => NHWC
        return [output, mean.reshape((N, G)), var.reshape((N, G))]
    else:
        if data_layout == "NHWC":
            x = np.transpose(x, (0, 4, 1, 2, 3))  # NDHWC => NCDHW
        N, C, D, H, W = x.shape
        G = groups
        x = x.reshape((N * G, -1))
        mean = np.mean(x, axis=1, keepdims=True)
        var = np.var(x, axis=1, keepdims=True)
        output = (x - mean) / np.sqrt(var + epsilon)
        output = output.reshape((N, C, D, H, W)) * scale.reshape(
            (-1, 1, 1, 1)
        ) + bias.reshape((-1, 1, 1, 1))
        if data_layout == "NHWC":
            output = np.transpose(output, (0, 2, 3, 4, 1))  # NCDHW => NDHWC
        return [output, mean.reshape((N, G)), var.reshape((N, G))]


def naive_residual_biasadd_layer_norm(
    x, residual, scale, bias, epsilon, groups, data_layout, activation
):
    x = x + residual
    out = naive_group_norm(x, scale, bias, epsilon, groups, data_layout)
    if activation == "silu":
        out[0] = F.silu(paddle.to_tensor(out[0])).numpy()
    return out


def add_group_norm_silu_static_wrapper(
    x, residual, scale, bias, epsilon, groups, data_layout="NHWC", activation=""
):
    helper = LayerHelper('add_group_norm_silu', **locals())
    mean_out = helper.create_variable_for_type_inference(
        dtype=x.dtype, stop_gradient=True
    )
    variance_out = helper.create_variable_for_type_inference(
        dtype=x.dtype, stop_gradient=True
    )

    inputs = {'x': x}
    if bias is not None:
        inputs['bias'] = bias
    if scale is not None:
        inputs['scale'] = scale
    if residual is not None:
        inputs['residual'] = residual

    # create output
    group_norm_out = helper.create_variable_for_type_inference(dtype=x.dtype)
    residual_out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type="add_group_norm_silu",
        inputs=inputs,
        outputs={
            "y": group_norm_out,
            "residual_out": residual_out,
            "mean": mean_out,
            "variance": variance_out,
        },
        attrs={
            "epsilon": epsilon,
            "groups": groups,
            "data_format": data_layout,
            "activation": activation,
        },
    )

    return group_norm_out, residual_out


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_float16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA or not support the bfloat16",
)
class TestGroupNormNHWC_StaticOp(unittest.TestCase):
    def setUp(self):
        np.random.seed(20)
        self.shape = (2, 4, 2, 6)
        self.r_shape = (1, 1, 1, 6)
        self.x_np = np.random.uniform(-0.05, 0.05, self.shape)
        self.residual_np = np.random.uniform(-0.05, 0.05, self.r_shape)
        self.scale_np = np.random.uniform(-0.05, 0.05, [self.shape[-1]])
        self.bias_np = np.random.uniform(-0.05, 0.05, [self.shape[-1]])
        self.epsilon = 1e-5
        self.groups = 2
        self.data_layout = 'NHWC'
        self.activation = ''
        self.place = paddle.CUDAPlace(0)

    def check_residual_add_groupnorm(
        self, x_np, scale_np, bias_np, residual_np, activation, dtype
    ):
        paddle.disable_static()
        navie_groupnorm_out = naive_residual_biasadd_layer_norm(
            x_np,
            residual_np,
            scale_np,
            bias_np,
            self.epsilon,
            self.groups,
            self.data_layout,
            self.activation,
        )
        navie_residual_out = naive_residual_add(x_np, residual_np)
        paddle.enable_static()

        with paddle.pir_utils.OldIrGuard():
            with paddle.static.program_guard(paddle.static.Program()):
                x_static = paddle.static.data(
                    name="x_static", shape=self.shape, dtype=dtype
                )
                residual_static = paddle.static.data(
                    name="residual_static",
                    shape=self.r_shape,
                    dtype=dtype,
                )

                scale_static = paddle.static.data(
                    name="scale_static", shape=[self.shape[-1]], dtype=dtype
                )
                bias_static = paddle.static.data(
                    name="bias_static", shape=[self.shape[-1]], dtype=dtype
                )
                outs = add_group_norm_silu_static_wrapper(
                    x_static,
                    residual_static,
                    scale_static,
                    bias_static,
                    self.epsilon,
                    self.groups,
                    self.data_layout,
                    activation,
                )

                exe = base.Executor(self.place)
                out_s = exe.run(
                    feed={
                        "x_static": x_np.astype(dtype),
                        "scale_static": scale_np.astype(dtype),
                        "residual_static": residual_np.astype(dtype),
                        "bias_static": bias_np.astype(dtype),
                    },
                    fetch_list=[outs],
                )
        return (out_s[0], out_s[1]), navie_groupnorm_out, navie_residual_out

    def test_residual_add_groupnorm_fp16(self):
        if not paddle.is_compiled_with_cuda():
            return
        self.dtype = np.float16
        (
            paddle_group_list,
            paddle_naive_group_out,
            paddle_naive_group_residual,
        ) = self.check_residual_add_groupnorm(
            self.x_np.astype(self.dtype),
            self.scale_np.astype(self.dtype),
            self.bias_np.astype(self.dtype),
            self.residual_np.astype(self.dtype),
            self.activation,
            self.dtype,
        )
        np.testing.assert_allclose(
            paddle_group_list[1],
            paddle_naive_group_residual,
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            paddle_group_list[0],
            paddle_naive_group_out[0],
            rtol=1e-4,
            atol=1e-4,
        )


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_float16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA or not support the bfloat16",
)
class TestGroupNormNHWCSilu_StaticOp(TestGroupNormNHWC_StaticOp):
    def setUp(self):
        np.random.seed(20)
        self.shape = (2, 4, 2, 6)
        self.r_shape = (1, 1, 1, 6)
        self.x_np = np.random.uniform(-0.05, 0.05, self.shape)
        self.residual_np = np.random.uniform(-0.05, 0.05, self.r_shape)
        self.scale_np = np.random.uniform(-0.05, 0.05, [self.shape[-1]])
        self.bias_np = np.random.uniform(-0.05, 0.05, [self.shape[-1]])
        self.epsilon = 1e-5
        self.groups = 2
        self.data_layout = 'NHWC'
        self.activation = 'silu'
        self.place = paddle.CUDAPlace(0)


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_float16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA or not support the bfloat16",
)
class TestGroupNormNHWC_StaticOp_1(TestGroupNormNHWC_StaticOp):
    def setUp(self):
        np.random.seed(20)
        self.shape = (2, 4, 2, 6)
        self.r_shape = (2, 4, 2, 6)
        self.x_np = np.random.uniform(-0.05, 0.05, self.shape)
        self.residual_np = np.random.uniform(-0.05, 0.05, self.r_shape)
        self.scale_np = np.random.uniform(-0.05, 0.05, [self.shape[-1]])
        self.bias_np = np.random.uniform(-0.05, 0.05, [self.shape[-1]])
        self.epsilon = 1e-5
        self.groups = 2
        self.data_layout = 'NHWC'
        self.activation = 'silu'
        self.place = paddle.CUDAPlace(0)


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_float16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA or not support the bfloat16",
)
class TestGroupNormNHWCSilu_StaticOp_1(TestGroupNormNHWC_StaticOp):
    def setUp(self):
        np.random.seed(20)
        self.shape = (2, 4, 2, 6)
        self.r_shape = (2, 4, 2, 6)
        self.x_np = np.random.uniform(-0.05, 0.05, self.shape)
        self.residual_np = np.random.uniform(-0.05, 0.05, self.r_shape)
        self.scale_np = np.random.uniform(-0.05, 0.05, [self.shape[-1]])
        self.bias_np = np.random.uniform(-0.05, 0.05, [self.shape[-1]])
        self.epsilon = 1e-5
        self.groups = 2
        self.data_layout = 'NHWC'
        self.activation = 'silu'
        self.place = paddle.CUDAPlace(0)


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_float16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA or not support the bfloat16",
)
class TestGroupNormNHWCSingleC_StaticOp(TestGroupNormNHWC_StaticOp):
    def setUp(self):
        np.random.seed(20)
        self.shape = (2, 4, 2, 6)
        self.r_shape = (2, 4, 2, 6)
        self.x_np = np.random.uniform(-0.05, 0.05, self.shape)
        self.residual_np = np.random.uniform(-0.05, 0.05, self.r_shape)
        self.scale_np = np.random.uniform(-0.05, 0.05, [self.shape[-1]])
        self.bias_np = np.random.uniform(-0.05, 0.05, [self.shape[-1]])
        self.epsilon = 1e-5
        self.groups = 6
        self.data_layout = 'NHWC'
        self.activation = ''
        self.place = paddle.CUDAPlace(0)


if __name__ == "__main__":
    unittest.main()
