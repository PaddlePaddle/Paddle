#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import unittest
import numpy as np
import struct

import paddle.fluid.core as core
from paddle.fluid.tests.unittests.op_test import OpTest, convert_float_to_uint16, OpTestTool
from paddle.fluid.tests.unittests.test_conv2d_op import conv2d_forward_naive, TestConv2DOp


def conv2d_residual_naive(out, residual):
    assert out.shape == residual.shape
    out = np.add(out, residual)
    return out


@unittest.skipIf(not core.supports_bfloat16(),
                 "place does not support BF16 evaluation")
class TestConv2DBF16Op(TestConv2DOp):
    def setUp(self):
        self.op_type = "conv2d"
        self.use_cudnn = False
        self.exhaustive_search = False
        self.use_cuda = False
        self.use_mkldnn = True
        self._cpu_only = True
        self.weight_type = np.float32
        self.input_type = np.float32
        self.mkldnn_data_type = "bfloat16"
        self.force_fp32_output = False
        self.init_group()
        self.init_dilation()
        self.init_test_case()
        self.init_fuse_relu()
        self.init_fuse_residual()
        self.init_data_type()
        self.init_force_fp32_output()
        self.init_infer_or_train()

        self.conv2d_param = {
            'stride': self.stride,
            'pad': self.pad,
            'dilation': self.dilations
        }

        self.input = np.random.random(self.input_size).astype(np.float32)
        self.filter = np.random.random(self.filter_size).astype(np.float32)

        self.inputs_fp32 = {'Input': self.input, 'Filter': self.filter}

        conv_out, _, _, _, _ = conv2d_forward_naive(
            self.input, self.filter, self.groups, self.conv2d_param)
        self.conv_output_float = conv_out

        if self.fuse_residual:
            self.input_residual = np.random.random(
                self.input_residual_size).astype(np.float32)
            self.conv_output_float = conv2d_residual_naive(
                self.conv_output_float, self.input_residual)
            self.conv_output = convert_float_to_uint16(self.conv_output_float)
            self.outputs = {'Output': self.conv_output}
        elif self.force_fp32_output:
            self.outputs = {'Output': self.conv_output_float.astype(np.float32)}
        else:
            self.outputs = {
                'Output': convert_float_to_uint16(self.conv_output_float)
            }

        if self.input_type is not np.float32:
            self.input = convert_float_to_uint16(self.input)

        if self.weight_type is not np.float32:
            self.filter = convert_float_to_uint16(self.filter)

        self.inputs = {
            'Input': self.input,
            'Filter': OpTest.np_dtype_to_fluid_dtype(
                self.filter.astype(self.weight_type))
        }

        if self.fuse_residual:
            self.inputs['ResidualData'] = OpTest.np_dtype_to_fluid_dtype(
                convert_float_to_uint16(self.input_residual))

        self.attrs = {
            'strides': self.stride,
            'paddings': self.pad,
            'groups': self.groups,
            'dilations': self.dilations,
            'use_cudnn': self.use_cudnn,
            'use_mkldnn': self.use_mkldnn,
            'mkldnn_data_type': self.mkldnn_data_type,
            'force_fp32_output': self.force_fp32_output,
            'fuse_residual_connection': self.fuse_residual
        }

        self.init_additional_attrs()

    def test_check_output(self):
        self.check_output_with_place(core.CPUPlace())

    def test_check_grad(self):
        pass

    def test_check_grad_no_filter(self):
        pass

    def test_check_grad_no_input(self):
        pass

    def init_test_case(self):
        TestConv2DOp.init_test_case(self)
        self.input_size = [1, 6, 12, 12]  # NCHW
        f_c = self.input_size[1] // self.groups
        o_c = 15
        self.input_residual_size = [1, o_c, 10, 10]
        self.filter_size = [o_c, f_c, 3, 3]

    def init_padding(self):
        pass

    def init_data_type(self):
        self.weight_type = np.float32
        self.input_type = np.uint16

    def init_force_fp32_output(self):
        self.force_fp32_output = False

    def init_fuse_relu(self):
        self.fuse_activation = "relu"

    def init_fuse_residual(self):
        self.fuse_residual = True

    def init_infer_or_train(self):
        self.weight_type = np.float32

    def init_additional_attrs(self):
        self.attrs['is_test'] = True


@OpTestTool.skip_if_not_cpu_bf16()
class TestConv2DWithGradBF16Op(TestConv2DBF16Op):
    def init_fuse_relu(self):
        self.fuse_activation = None

    def init_fuse_residual(self):
        self.fuse_residual = None

    def init_additional_attrs(self):
        self.attrs['is_test'] = False

    def init_infer_or_train(self):
        self.weight_type = np.uint16

    def test_check_grad(self):
        dout = self.conv_output_float
        x = self.inputs_fp32['Input']
        w = self.inputs_fp32['Filter']

        dx, dweights = conv_backward(dout, x, w, self.conv2d_param)

        self.check_grad_with_place(
            core.CPUPlace(), ["Input", "Filter"],
            "Output",
            user_defined_grads=[dx, dweights],
            user_defined_grad_outputs=[convert_float_to_uint16(dout)])

    def test_check_grad_no_filter(self):
        dout = self.conv_output_float
        x = self.inputs_fp32['Input']
        w = self.inputs_fp32['Filter']

        dx, _ = conv_backward(dout, x, w, self.conv2d_param)

        self.check_grad_with_place(
            core.CPUPlace(), ["Input"],
            "Output",
            set(['Filter']),
            user_defined_grads=[dx],
            user_defined_grad_outputs=[convert_float_to_uint16(dout)])

    def test_check_grad_no_input(self):
        dout = self.conv_output_float
        x = self.inputs_fp32['Input']
        w = self.inputs_fp32['Filter']

        _, dweights = conv_backward(dout, x, w, self.conv2d_param)

        self.check_grad_with_place(
            core.CPUPlace(), ["Filter"],
            "Output",
            set(['Input']),
            user_defined_grads=[dweights],
            user_defined_grad_outputs=[convert_float_to_uint16(dout)])


def conv_backward(dout, x, w, params):
    padding = params['pad'][0]
    stride = params['stride']

    dx = np.zeros_like(x)
    dweights = np.zeros_like(w)

    N, IC, H, W = x.shape
    OC, _, KH, KW = w.shape

    H_out = int(1 + (H + 2 * padding - KH) / stride[0])
    W_out = int(1 + (W + 2 * padding - KW) / stride[1])

    x_padded = np.pad(x, ((0, ), (0, ), (padding, ), (padding, )), 'constant')

    for n in range(N):
        for oc in range(OC):
            for i in range(KH):
                for j in range(KW):
                    for k in range(H_out):
                        for l in range(W_out):
                            for ic in range(IC):
                                dweights[oc, ic, i, j] += x_padded[
                                    n, ic, i + k * stride[0], j + l * stride[
                                        1]] * dout[n, oc, k, l]

    dx_padded = np.pad(dx, ((0, ), (0, ), (padding, ), (padding, )), 'constant')

    w_ = np.zeros_like(w)
    for i in range(KH):
        for j in range(KW):
            w_[:, :, i, j] = w[:, :, KH - i - 1, KW - j - 1]

    for n in range(N):
        for oc in range(OC):
            for i in range(H_out):
                for j in range(W_out):
                    for kh in range(KH):
                        for kw in range(KW):
                            for ic in range(IC):
                                dx_padded[n, ic, stride[0] * i + kh, stride[1] *
                                          j + kw] += dout[n, oc, i, j] * w[
                                              oc, ic, kh, kw]

    if padding == 0:
        dx = dx_padded
    else:
        dx = dx_padded[:, :, padding:-padding, padding:-padding]

    return dx.astype(np.float32), dweights.astype(np.float32)


class TestConv2DBF16WithPadding1(TestConv2DWithGradBF16Op):
    def init_test_case(self):
        TestConv2DWithGradBF16Op.init_test_case(self)
        self.pad = [1, 1]


class TestConv2DBF16WithStride2(TestConv2DWithGradBF16Op):
    def init_test_case(self):
        TestConv2DWithGradBF16Op.init_test_case(self)
        self.stride = [2, 3]


class TestConv2D(TestConv2DBF16Op):
    def init_test_case(self):
        self.pad = [0, 0]
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]  # NCHW
        self.input_residual_size = [2, 6, 3, 3]
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 3]

    def init_data_type(self):
        self.input_type = np.uint16


class TestWithPad(TestConv2D):
    def init_test_case(self):
        TestConv2D.init_test_case(self)
        self.pad = [1, 1]
        self.input_residual_size = [2, 6, 5, 5]


class TestWithGroup(TestConv2D):
    def init_group(self):
        self.groups = 3


class TestWithStride(TestConv2DBF16Op):
    def init_test_case(self):
        self.pad = [1, 1]
        self.stride = [2, 2]
        self.input_size = [2, 3, 6, 6]
        self.input_residual_size = [2, 6, 3, 3]
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 3]

    def init_data_type(self):
        self.input_type = np.uint16


class TestWithDilations(TestConv2DBF16Op):
    def init_test_case(self):
        self.pad = [1, 1]
        self.stride = [1, 1]
        self.dilations = [2, 2]
        self.input_size = [2, 3, 10, 10]
        self.input_residual_size = [2, 6, 8, 8]
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 3]

    def init_data_type(self):
        self.input_type = np.uint16


class TestWith1x1ForceFP32Output(TestConv2DBF16Op):
    def init_test_case(self):
        self.pad = [0, 0]
        self.stride = [1, 1]
        self.input_size = [1, 3, 5, 5]
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 1, 1]

    def init_force_fp32_output(self):
        self.force_fp32_output = True

    def init_fuse_residual(self):
        self.fuse_residual = False


class TestWithInput1x1Filter1x1(TestConv2DBF16Op):
    def init_test_case(self):
        self.pad = [0, 0]
        self.stride = [1, 1]
        self.input_size = [2, 3, 1, 1]
        self.input_residual_size = [2, 6, 1, 1]
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 1, 1]

    def init_group(self):
        self.groups = 3


if __name__ == '__main__':
    from paddle import enable_static
    enable_static()
    unittest.main()
