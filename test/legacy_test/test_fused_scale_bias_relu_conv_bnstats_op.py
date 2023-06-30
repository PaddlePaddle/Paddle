#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from eager_op_test import OpTest, skip_check_grad_ci

import paddle
from paddle import nn
from paddle.fluid import core, framework
from paddle.fluid.executor import Executor


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
class TestFusedScaleBiasReluConvBnstatsOp(OpTest):
    def setUp(self):
        self.__class__.op_type = "fused_scale_bias_relu_conv_bnstats"
        self.dtype = np.float16
        self.math_type = np.float32
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

    def has_cuda(self):
        return core.is_compiled_with_cuda()

    def get_feed_map(self, inputs, place):
        feed_map = {}
        for name in inputs:
            tensor = core.LoDTensor()
            tensor.set(inputs[name], place)
            feed_map[name] = tensor
        return feed_map

    def calc_normal_pass(self):
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
        return (
            after_conv.numpy().astype(self.dtype),
            after_bn.numpy().astype(self.dtype),
            self.bn._mean.numpy(),
            self.bn._variance.numpy(),
            mean_np,
            invstd_np,
        )

    def calc_fused_pass(self, place):
        paddle.enable_static()
        program = framework.Program()
        block = program.global_block()

        x_var = block.create_var(name="x", shape=self.x_size, dtype='float16')
        w_var = block.create_var(
            name="w", shape=self.filter_size, dtype='float16'
        )
        scale_var = block.create_var(
            name="scale", shape=self.scale_size, dtype='float16'
        )
        bias_var = block.create_var(
            name="bias", shape=self.scale_size, dtype='float16'
        )
        y_var = block.create_var(name="out", dtype='float16')
        bn_scale = block.create_var(
            name="bn_scale", shape=self.bn_size, dtype='float32'
        )
        bn_bias = block.create_var(
            name="bn_bias", shape=self.bn_size, dtype='float32'
        )
        bn_running_mean = block.create_var(
            name="input_running_mean", shape=self.bn_size, dtype='float32'
        )
        bn_running_var = block.create_var(
            name="input_running_var", shape=self.bn_size, dtype='float32'
        )
        updated_running_mean = block.create_var(
            name="out_running_mean", shape=self.bn_size, dtype='float32'
        )
        updated_running_var = block.create_var(
            name="out_running_var", shape=self.bn_size, dtype='float32'
        )
        saved_mean = block.create_var(
            name="saved_mean", shape=self.bn_size, dtype='float32'
        )
        saved_inv_var = block.create_var(
            name="saved_var", shape=self.bn_size, dtype='float32'
        )
        eq_scale = block.create_var(
            name="eq_scale", shape=self.bn_size, dtype='float16'
        )
        eq_bias = block.create_var(
            name="eq_bias", shape=self.bn_size, dtype='float16'
        )
        op_inputs = {
            'x': x_var,
            'w': w_var,
            'scale': scale_var,
            'bias': bias_var,
            'bn_scale': bn_scale,
            'bn_bias': bn_bias,
            'input_running_mean': bn_running_mean,
            'input_running_var': bn_running_var,
        }
        op_outputs = {
            'out': y_var,
            'out_running_mean': updated_running_mean,
            'out_running_var': updated_running_var,
            'saved_mean': saved_mean,
            'saved_var': saved_inv_var,
            'eq_scale': eq_scale,
            'eq_bias': eq_bias,
        }
        op = block.append_op(
            type=self.__class__.op_type,
            inputs=op_inputs,
            outputs=op_outputs,
            attrs=self.attrs,
        )
        op.desc.infer_shape(block.desc)

        # execute program
        feed_map = self.get_feed_map(self.inputs, place)
        fetch_list = [
            'out',
            'eq_scale',
            'eq_bias',
            'out_running_mean',
            'out_running_var',
            'saved_mean',
            'saved_var',
        ]

        executor = Executor(place)
        outs = executor.run(
            program, feed=feed_map, fetch_list=fetch_list, return_numpy=True
        )
        (
            y_out,
            eq_scale_out,
            eq_bias_out,
            running_mean_out,
            running_var_out,
            saved_mean_out,
            saved_invvar_out,
        ) = outs
        bn_out = y_out * eq_scale_out.reshape(
            [1, 1, 1, -1]
        ) + eq_bias_out.reshape([1, 1, 1, -1])
        return (
            y_out,
            bn_out,
            running_mean_out,
            running_var_out,
            saved_mean_out,
            saved_invvar_out,
        )

    def test_check_output(self):
        if self.has_cuda():
            place = core.CUDAPlace(0)
            (
                y_ref,
                bn_out_ref,
                running_mean_out_ref,
                running_var_out_ref,
                saved_mean_out_ref,
                saved_invvar_out_ref,
            ) = self.calc_normal_pass()
            (
                y_out,
                bn_out,
                running_mean_out,
                running_var_out,
                saved_mean_out,
                saved_invvar_out,
            ) = self.calc_fused_pass(place)

            np.testing.assert_allclose(
                y_ref, y_out, rtol=self.rtol, atol=self.atol
            )

            np.testing.assert_allclose(
                bn_out_ref, bn_out, rtol=self.rtol, atol=self.atol
            )
            np.testing.assert_allclose(
                running_mean_out_ref,
                running_mean_out,
                rtol=self.rtol,
                atol=self.atol,
            )
            np.testing.assert_allclose(
                running_var_out_ref,
                running_var_out,
                rtol=self.rtol,
                atol=self.atol,
            )
            np.testing.assert_allclose(
                saved_mean_out_ref,
                saved_mean_out,
                rtol=self.rtol,
                atol=self.atol,
            )
            np.testing.assert_allclose(
                saved_invvar_out_ref,
                saved_invvar_out,
                rtol=self.rtol,
                atol=self.atol,
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


@skip_check_grad_ci(reason="no grap op")
@unittest.skipIf(skip_unit_test(), skip_msg)
class TestFusedScaleBiasReluConvBnstatsOpNoPrologue(
    TestFusedScaleBiasReluConvBnstatsOp
):
    def init_attr(self):
        self.fuse_prologue = False
        self.exhaustive_search = False


@skip_check_grad_ci(reason="no grap op")
@unittest.skipIf(skip_unit_test(), skip_msg)
class TestFusedScaleBiasReluConvBnstatsOpExhaustive(
    TestFusedScaleBiasReluConvBnstatsOp
):
    def init_attr(self):
        self.fuse_prologue = True
        self.exhaustive_search = True


if __name__ == '__main__':
    unittest.main()
