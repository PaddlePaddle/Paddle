#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest, skip_check_grad_ci

import paddle
from paddle import nn
from paddle.base import core


def skip_unit_test():
    return (
        not paddle.is_compiled_with_cuda()
        or paddle.device.cuda.get_device_capability()[0] < 8
        or paddle.get_cudnn_version() < 8800
    )


skip_msg = (
    "only support with cuda and CUDNN 8.8 or later,"
    " and only Ampere or later devices are supported"
)


@skip_check_grad_ci(reason="no grap op")
@unittest.skipIf(skip_unit_test(), skip_msg)
class TestFusedScaleBiasReluConvBnOp(OpTest):
    def setUp(self):
        self.__class__.op_type = "fused_scale_bias_relu_conv_bn"
        self.dtype = np.float16
        self.outputs = None
        self.padding_algorithm = "EXIPLICIT"
        self.data_format = "NHWC"
        self.groups = 1
        self.init_attr()
        self.init_test_case()
        self.rtol = 1e-5
        self.atol = 2e-2

        self.attrs = {
            'fuse_prologue': self.fuse_prologue,
            'strides': self.stride,
            'paddings': self.pad,
            'dilations': self.dilations,
            'data_format': self.data_format,
            'padding_algorithm': self.padding_algorithm,
            'accumulation_count': self.accumulation_count,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'exhaustive_search': self.exhaustive_search,
            'groups': self.groups,
        }

        # prepare inputs
        np.random.seed(0)
        self.x_input = np.random.random(self.x_size).astype(self.dtype)
        self.bias_input = np.random.random(self.in_channel_num).astype(
            self.dtype
        )
        self.scale_input = np.random.random(self.in_channel_num).astype(
            self.dtype
        )

        self.x_input_prologue = self.x_input.astype(np.float32)
        if self.fuse_prologue:
            self.x_input_prologue *= self.scale_input.reshape(
                (1, 1, 1, self.in_channel_num)
            ).astype(
                np.float32
            )  # scale
            self.x_input_prologue += self.bias_input.reshape(
                (1, 1, 1, self.in_channel_num)
            ).astype(
                np.float32
            )  # bias
            self.x_input_prologue = np.maximum(self.x_input_prologue, 0)  # relu
        self.x_input_prologue = self.x_input_prologue.astype(self.dtype)

        paddle.disable_static()
        paddle.seed(0)
        paddle.set_default_dtype(self.dtype)

        self.conv = nn.Conv2D(
            in_channels=self.x_size[-1],
            out_channels=self.filter_size[0],
            kernel_size=self.filter_size[-1],
            stride=self.stride,
            padding=self.pad,
            groups=self.groups,
            bias_attr=False,
            data_format=self.data_format,
        )

        self.bn = nn.BatchNorm(
            self.filter_size[0],
            momentum=self.momentum,
            epsilon=self.epsilon,
            data_layout=self.data_format,
        )

        self.w_input = self.conv.weight.numpy().astype(self.dtype)
        self.bn_scale_input = self.bn.weight.numpy()
        self.bn_bias_input = self.bn.bias.numpy()
        self.bn_running_mean_input = self.bn._mean.numpy()
        self.bn_running_var_input = self.bn._variance.numpy()

        (
            y_ref,
            running_mean_out_ref,
            running_var_out_ref,
            saved_mean_out_ref,
            saved_invvar_out_ref,
            eqscale_ref,
            eqbias_ref,
        ) = self.calc_ref()

        self.inputs = {
            'x': self.x_input,
            'w': self.w_input,
            'bn_scale': self.bn_scale_input,
            'bn_bias': self.bn_bias_input,
            'input_running_mean': self.bn_running_mean_input,
            'input_running_var': self.bn_running_var_input,
        }
        if self.fuse_prologue:
            extra_inputs = {
                'bias': self.bias_input,
                'scale': self.scale_input,
            }
            self.inputs.update(extra_inputs)

        self.outputs = {
            'out': y_ref,
            'out_running_mean': running_mean_out_ref,
            'out_running_var': running_var_out_ref,
            'saved_mean': saved_mean_out_ref,
            'saved_var': saved_invvar_out_ref,
            'eq_scale': eqscale_ref,
            'eq_bias': eqbias_ref,
        }

    def calc_ref(self):
        # Calculate normal (scale + bias + relu +) Conv + BN
        x_input_np = self.x_input
        if self.fuse_prologue:
            x_input_np = self.x_input_prologue
        x_tensor = paddle.to_tensor(x_input_np, stop_gradient=False)
        after_conv = self.conv(x_tensor)
        after_bn = self.bn(after_conv)
        # Calculate reference for saved_mean and saved_invvar
        after_conv_np = (
            after_conv.numpy()
            .astype(np.float32)
            .reshape((-1, after_conv.shape[-1]))
        )
        mean_np = after_conv_np.mean(axis=0)
        var_np = after_conv_np.var(axis=0)
        invstd_np = 1 / np.sqrt(var_np + self.epsilon)
        # Calculate reference for eqscale and eqbias
        eqscale_np = self.bn_scale_input * invstd_np
        eqbias_np = (
            self.bn_bias_input - self.bn_scale_input * mean_np * invstd_np
        )
        return (
            after_conv.numpy().astype(self.dtype),
            self.bn._mean.numpy(),
            self.bn._variance.numpy(),
            mean_np,
            invstd_np,
            eqscale_np,
            eqbias_np,
        )

    def has_cuda(self):
        return core.is_compiled_with_cuda()

    def test_check_output(self):
        if self.has_cuda():
            place = core.CUDAPlace(0)
            self.check_output_with_place(
                place, atol=self.atol, rtol=self.rtol, check_dygraph=False
            )

    def init_test_case(self):
        self.pad = [0, 0]
        self.stride = [1, 1]
        self.dilations = [1, 1]

        self.x_size = [8, 16, 16, 32]  # NHWC
        self.filter_size = [64, 32, 1, 1]
        self.y_size = [8, 16, 16, 64]
        self.in_channel_num = self.x_size[-1]
        self.out_channel_num = self.y_size[-1]
        self.scale_size = [self.in_channel_num]
        self.bn_size = [self.out_channel_num]
        self.momentum = 0.9
        self.epsilon = 1e-5
        self.accumulation_count = (
            self.y_size[0] * self.y_size[1] * self.y_size[2]
        )

    def init_attr(self):
        self.fuse_prologue = True
        self.exhaustive_search = False


class TestFusedScaleBiasReluConvBnOpNoPrologue(TestFusedScaleBiasReluConvBnOp):
    def init_attr(self):
        self.fuse_prologue = False
        self.exhaustive_search = False


class TestFusedScaleBiasReluConvBnOpExhaustive(TestFusedScaleBiasReluConvBnOp):
    def init_attr(self):
        self.fuse_prologue = True
        self.exhaustive_search = True


if __name__ == '__main__':
    unittest.main()
