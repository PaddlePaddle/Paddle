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
from paddle.base import core, framework
from paddle.base.executor import Executor


def skip_unit_test():
    return (
        not paddle.is_compiled_with_cuda()
        or paddle.device.cuda.get_device_capability()[0] < 8
    )


skip_msg = "only support with cuda and Ampere or later devices"


@skip_check_grad_ci(reason="no grad op")
@unittest.skipIf(skip_unit_test(), skip_msg)
class TestFusedDconvDreluDbnOp(OpTest):
    def setUp(self):
        self.__class__.op_type = "fused_dconv_drelu_dbn"
        self.dtype = np.float16
        self.math_type = np.float32
        self.outputs = None
        self.padding_algorithm = "EXIPLICIT"
        self.data_format = "NHWC"
        self.groups = 1
        self.rtol = 1e-5
        self.atol = 2e-2

        self.init_dilation()
        self.init_test_case()
        self.init_paddings()
        self.init_attr()

        self.X1 = np.random.random(self.input_size).astype(self.dtype) - 0.5
        self.X2 = np.random.random(self.input_size).astype(self.dtype) - 0.5
        self.dY1 = np.random.random(self.output_size).astype(self.dtype) - 0.5
        self.dY2 = np.random.random(self.input_size).astype(self.dtype) - 0.5

        paddle.disable_static()
        paddle.seed(0)
        paddle.set_default_dtype(self.dtype)
        self.bn1 = nn.BatchNorm(
            self.input_size[-1],
            momentum=self.momentum,
            epsilon=self.epsilon,
            data_layout=self.data_format,
        )
        self.bn2 = nn.BatchNorm(
            self.input_size[-1],
            momentum=self.momentum,
            epsilon=self.epsilon,
            data_layout=self.data_format,
        )
        self.relu = nn.ReLU()
        self.conv = nn.Conv2D(
            in_channels=self.input_size[-1],
            out_channels=self.filter_size[0],
            kernel_size=self.filter_size[-1],
            stride=self.stride,
            padding=self.pad,
            groups=1,
            bias_attr=False,
            data_format=self.data_format,
        )

        self.w_input = self.conv.weight.numpy().astype(self.dtype)
        self.bn1_scale_input = self.bn1.weight.numpy()
        self.bn1_bias_input = self.bn1.bias.numpy()
        self.bn1_running_mean_input = self.bn1._mean.numpy()
        self.bn1_running_var_input = self.bn1._variance.numpy()

        self.bn2_scale_input = self.bn2.weight.numpy()
        self.bn2_bias_input = self.bn2.bias.numpy()
        self.bn2_running_mean_input = self.bn2._mean.numpy()
        self.bn2_running_var_input = self.bn2._variance.numpy()

    def has_cuda(self):
        return core.is_compiled_with_cuda()

    def get_feed_map(self, inputs, place):
        feed_map = {}
        for name in inputs:
            tensor = core.DenseTensor()
            tensor.set(inputs[name], place)
            feed_map[name] = tensor
        return feed_map

    def calc_normal_pass(self):
        """
        Given dY, get dX for the following pattern:
        (1) X1 -> BN1 -> ReLU -> Conv -> Y
        (2) with fuse_dual = True:
            X1 -> BN1 -> Add -> ReLU -> Conv -> Y
            X2 -> BN2 ---/
        (3) with fuse_shortcut = True:
            X1 -> BN1 -> Add -> ReLU -> Conv -> Y
            X2 ----------/
        (4) with fuse_add = True:
                               /-------> Y2
            X1 -> BN1 -> ReLU -> Conv -> Y
        fuse_add is also compatible with case (2) and (3)
        """
        # inputs
        x1_tensor = paddle.to_tensor(self.X1, stop_gradient=False)
        x2_tensor = paddle.to_tensor(self.X2, stop_gradient=False)
        dy1_tensor = paddle.to_tensor(self.dY1, stop_gradient=False)
        dy2_tensor = paddle.to_tensor(self.dY2, stop_gradient=False)

        if self.fuse_dual:
            before_relu = self.bn1(x1_tensor) + self.bn2(x2_tensor)
        elif self.fuse_shortcut:
            before_relu = self.bn1(x1_tensor) + x2_tensor
        else:
            before_relu = self.bn1(x1_tensor)

        after_relu = self.relu(before_relu)
        y1_tensor = self.conv(after_relu)
        y2_tensor = after_relu * 1

        if self.fuse_add:
            paddle.autograd.backward(
                [y1_tensor, y2_tensor], [dy1_tensor, dy2_tensor], True
            )
        else:
            paddle.autograd.backward([y1_tensor], [dy1_tensor], True)

        self.conv_x = after_relu.numpy()
        # ['dW', 'dX1', "BN1_dGamma", "BN1_dBeta"]
        outputs = [
            self.conv.weight.grad.numpy(),
            x1_tensor.grad.numpy(),
            self.bn1.weight.grad.numpy(),
            self.bn1.bias.grad.numpy(),
        ]
        if self.fuse_dual or self.fuse_shortcut:
            # ['dX2']
            outputs.append(x2_tensor.grad.numpy())
        if self.fuse_dual:
            # ['BN2_dGamma', 'BN1_dBeta']
            outputs += [
                self.bn2.weight.grad.numpy(),
                self.bn2.bias.grad.numpy(),
            ]
        return outputs

    def _calc_mean_invstd(
        self,
        input,
        bn_scale_np,
        bn_bias_np,
    ):
        input = input.astype(self.math_type).reshape((-1, input.shape[-1]))
        sample_mean = input.mean(axis=0)
        sample_var = input.var(axis=0)
        sample_invstd = 1 / np.sqrt(sample_var + self.epsilon)
        sample_eqscale = bn_scale_np * sample_invstd
        sample_eqbias = -bn_scale_np * sample_invstd * sample_mean + bn_bias_np
        return (
            sample_mean,
            sample_invstd,
            sample_eqscale.astype(self.dtype),
            sample_eqbias.astype(self.dtype),
        )

    def calc_mean_invstd(self, place):
        (
            self.bn1_saved_mean,
            self.bn1_saved_invstd,
            self.bn1_eqscale,
            self.bn1_eqbias,
        ) = self._calc_mean_invstd(
            self.X1,
            self.bn1_scale_input,
            self.bn1_bias_input,
        )

        (
            self.bn2_saved_mean,
            self.bn2_saved_invstd,
            _,
            _,
        ) = self._calc_mean_invstd(
            self.X2,
            self.bn2_scale_input,
            self.bn2_bias_input,
        )

    def calc_fused_pass(self, place):
        self.calc_mean_invstd(place)

        paddle.enable_static()
        program = framework.Program()
        block = program.global_block()
        bn_size = [self.input_size[-1]]

        dY1 = block.create_var(
            name="dY1", shape=self.output_size, dtype='float16'
        )
        dY2 = block.create_var(
            name="dY2", shape=self.input_size, dtype='float16'
        )
        W = block.create_var(name="W", shape=self.filter_size, dtype='float16')
        dW = block.create_var(
            name="dW", shape=self.filter_size, dtype='float16'
        )
        X1 = block.create_var(name="X1", shape=self.input_size, dtype='float16')
        X2 = block.create_var(name="X2", shape=self.input_size, dtype='float16')
        Conv_X = block.create_var(
            name="Conv_X", shape=self.input_size, dtype='float16'
        )
        BN1_mean = block.create_var(
            name="BN1_mean", shape=bn_size, dtype='float32'
        )
        BN1_inv_std = block.create_var(
            name="BN1_inv_std", shape=bn_size, dtype='float32'
        )
        BN1_scale = block.create_var(
            name="BN1_scale", shape=bn_size, dtype='float32'
        )
        BN1_bias = block.create_var(
            name="BN1_bias", shape=bn_size, dtype='float32'
        )
        BN1_eqscale = block.create_var(
            name="BN1_eqscale", shape=bn_size, dtype='float16'
        )
        BN1_eqbias = block.create_var(
            name="BN1_eqbias", shape=bn_size, dtype='float16'
        )
        BN2_mean = block.create_var(
            name="BN2_mean", shape=bn_size, dtype='float32'
        )
        BN2_inv_std = block.create_var(
            name="BN2_inv_std", shape=bn_size, dtype='float32'
        )
        BN2_scale = block.create_var(
            name="BN2_scale", shape=bn_size, dtype='float32'
        )
        BN2_bias = block.create_var(
            name="BN2_bias", shape=bn_size, dtype='float32'
        )
        # outputs
        BN1_dGamma = block.create_var(
            name="BN1_dGamma", shape=bn_size, dtype='float32'
        )
        BN1_dBeta = block.create_var(
            name="BN1_dBeta", shape=bn_size, dtype='float32'
        )
        BN2_dGamma = block.create_var(
            name="BN2_dGamma", shape=bn_size, dtype='float32'
        )
        BN2_dBeta = block.create_var(
            name="BN2_dBeta", shape=bn_size, dtype='float32'
        )
        dX1 = block.create_var(
            name="dX1", shape=self.input_size, dtype='float16'
        )
        dX2 = block.create_var(
            name="dX2", shape=self.input_size, dtype='float16'
        )

        op_attrs = {
            'strides': self.stride,
            'paddings': self.pad,
            'dilations': self.dilations,
            'padding_algorithm': self.padding_algorithm,
            'groups': self.groups,
            'data_format': self.data_format,
            'fuse_shortcut': self.fuse_shortcut,
            'fuse_dual': self.fuse_dual,
            'fuse_add': self.fuse_add,
            'exhaustive_search': self.exhaustive_search,
        }

        op_inputs = {
            'grad_output': dY1,
            'weight': W,
            'bn1_mean': BN1_mean,
            'bn1_inv_std': BN1_inv_std,
            'bn1_gamma': BN1_scale,
            'bn1_beta': BN1_bias,
            'bn1_input': X1,
        }

        op_outputs = {
            'grad_bn1_input': dX1,
            'grad_bn1_gamma': BN1_dGamma,
            'grad_bn1_beta': BN1_dBeta,
            'grad_weight': dW,
        }

        if self.fuse_add:
            op_inputs['grad_output_add'] = dY2

        if self.fuse_shortcut:
            op_inputs['residual_input'] = X2
            op_outputs['grad_bn2_input'] = dX2

        if self.fuse_dual:
            extra_inputs = {
                'bn2_mean': BN2_mean,
                'bn2_inv_std': BN2_inv_std,
                'bn2_gamma': BN2_scale,
                'bn2_beta': BN2_bias,
                'bn2_input': X2,
            }
            op_inputs.update(extra_inputs)

            extra_outputs = {
                'grad_bn2_input': dX2,
                'grad_bn2_gamma': BN2_dGamma,
                'grad_bn2_beta': BN2_dBeta,
            }
            op_outputs.update(extra_outputs)

        if self.fuse_shortcut or self.fuse_dual:
            op_inputs['conv_input'] = Conv_X
        else:
            op_inputs['bn1_eqscale'] = BN1_eqscale
            op_inputs['bn1_eqbias'] = BN1_eqbias

        op = block.append_op(
            type=self.__class__.op_type,
            inputs=op_inputs,
            outputs=op_outputs,
            attrs=op_attrs,
        )

        # execute program
        graph_inputs = {
            'dY1': self.dY1,
            'dY2': self.dY2,
            'W': self.w_input,
            'X1': self.X1,
            'X2': self.X2,
            'BN1_mean': self.bn1_saved_mean,
            'BN1_inv_std': self.bn1_saved_invstd,
            'BN1_scale': self.bn1_scale_input,
            'BN1_bias': self.bn1_bias_input,
            'Conv_X': self.conv_x,
            'BN1_eqscale': self.bn1_eqscale,
            'BN1_eqbias': self.bn1_eqbias,
            'BN2_mean': self.bn2_saved_mean,
            'BN2_inv_std': self.bn2_saved_invstd,
            'BN2_scale': self.bn2_scale_input,
            'BN2_bias': self.bn2_bias_input,
        }

        feed_map = self.get_feed_map(graph_inputs, place)
        fetch_list = ['dW', 'dX1', "BN1_dGamma", "BN1_dBeta"]
        if self.fuse_dual or self.fuse_shortcut:
            fetch_list += ['dX2']
        if self.fuse_dual:
            fetch_list += ['BN2_dGamma', 'BN1_dBeta']

        executor = Executor(place)
        outs = executor.run(
            program, feed=feed_map, fetch_list=fetch_list, return_numpy=True
        )
        return outs, fetch_list

    def test_check_output(self):
        if self.has_cuda():
            place = core.CUDAPlace(0)
            outputs_expected = self.calc_normal_pass()
            outputs_actual, _ = self.calc_fused_pass(place)

            assert len(outputs_expected) == len(outputs_actual)
            for expected, actual in zip(outputs_expected, outputs_actual):
                np.testing.assert_allclose(
                    expected,
                    actual,
                    rtol=self.rtol,
                    atol=self.atol,
                )

    def init_test_case(self):
        self.pad = [0, 0]
        self.stride = [1, 1]
        self.input_size = [2, 5, 5, 16]  # NHWC
        self.output_size = [2, 5, 5, 32]
        assert np.mod(self.input_size[-1], self.groups) == 0
        f_c = self.input_size[-1] // self.groups
        self.filter_size = [32, f_c, 1, 1]
        self.momentum = 0.9
        self.epsilon = 1e-5
        self.accumulation_count = (
            self.input_size[0] * self.input_size[1] * self.input_size[2]
        )

    def init_dilation(self):
        self.dilations = [1, 1]

    def init_paddings(self):
        self.pad = [0, 0]
        self.padding_algorithm = "EXPLICIT"

    def init_attr(self):
        self.fuse_add = False
        self.fuse_shortcut = False
        self.fuse_dual = False
        self.exhaustive_search = False


@skip_check_grad_ci(reason="no grap op")
@unittest.skipIf(skip_unit_test(), skip_msg)
class TestFusedDconvDreluDbnOpShortcut(TestFusedDconvDreluDbnOp):
    def init_attr(self):
        self.fuse_add = False
        self.fuse_shortcut = True
        self.fuse_dual = False
        self.exhaustive_search = False


@skip_check_grad_ci(reason="no grap op")
@unittest.skipIf(skip_unit_test(), skip_msg)
class TestFusedDconvDreluDbnOpDual(TestFusedDconvDreluDbnOp):
    def init_attr(self):
        self.fuse_add = False
        self.fuse_shortcut = False
        self.fuse_dual = True
        self.exhaustive_search = False


@skip_check_grad_ci(reason="no grap op")
@unittest.skipIf(skip_unit_test(), skip_msg)
class TestFusedDconvDreluDbnOpShortcutAdd(TestFusedDconvDreluDbnOp):
    def init_attr(self):
        self.fuse_add = True
        self.fuse_shortcut = True
        self.fuse_dual = False
        self.exhaustive_search = False


@skip_check_grad_ci(reason="no grap op")
@unittest.skipIf(skip_unit_test(), skip_msg)
class TestFusedDconvDreluDbnOpDualAdd(TestFusedDconvDreluDbnOp):
    def init_attr(self):
        self.fuse_add = True
        self.fuse_shortcut = False
        self.fuse_dual = True
        self.exhaustive_search = False


@skip_check_grad_ci(reason="no grap op")
@unittest.skipIf(skip_unit_test(), skip_msg)
class TestFusedDconvDreluDbnOpExhaustive(TestFusedDconvDreluDbnOp):
    def init_attr(self):
        self.fuse_add = False
        self.fuse_shortcut = False
        self.fuse_dual = False
        self.exhaustive_search = True


if __name__ == '__main__':
    np.random.seed(0)
    unittest.main()
