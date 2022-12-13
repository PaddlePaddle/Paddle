# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import paddle.fluid.core as core
from paddle import enable_static
from paddle.fluid.tests.unittests.op_test import OpTest
from paddle.fluid.tests.unittests.test_conv2d_transpose_op import (
    conv2dtranspose_forward_naive,
)


def conv2d_bias_naive(out, bias):
    _, out_c, _, _ = out.shape

    for l in range(out_c):
        out[:, l, :, :] = out[:, l, :, :] + bias[l]
    return out


@unittest.skipIf(
    not core.supports_int8(), "place does not support int8 computation"
)
class TestConv2DTransposeINT8MKLDNNOp(OpTest):
    def test_check_output(self):
        self.check_output_with_place(core.CPUPlace())

    def test_check_grad(self):
        pass

    def test_check_grad_no_input(self):
        pass

    def test_check_grad_no_filter(self):
        pass

    def init_op_type(self):
        self.data_format = "NCHW"
        self.op_type = 'conv2d_transpose'
        self._cpu_only = True

    def init_test_case(self):
        self.pad = [0, 0]
        self.fuse_bias = False
        self.use_mkldnn = True
        self.is_test = True
        self.bias_size = None
        self.fuse_activation = ""
        self.fuse_alpha = 0.0
        self.fuse_beta = 0.0
        self.stride = [1, 1]
        self.dilations = [1, 1]
        self.input_size = [2, 3, 5, 5]  # NCHW
        f_c = self.input_size[1]
        self.filter_size = [f_c, 6, 3, 3]
        self.scale_in = 0.95
        self.scale_out = 0.5
        self.scale_weights = [10.0]
        self.groups = 1
        self.output_size = None
        self.output_padding = []
        self.pad = [0, 0]
        self.padding_algorithm = "EXPLICIT"
        self.force_fp32_output = False

    def setUp(self):
        self.input_type = np.int8
        self.dtype = np.int8
        self.srctype = np.uint8
        self.dsttype = np.int8
        self.mkldnn_data_type = "int8"
        self.weighttype = np.float32
        self.init_op_type()
        self.init_test_case()

        input = np.random.random(self.input_size).astype(np.uint8)
        # filter = np.random.random(self.filter_size).astype(np.float32)

        self.attrs = {
            'strides': self.stride,
            'paddings': self.pad,
            'padding_algorithm': self.padding_algorithm,
            'groups': self.groups,
            'dilations': self.dilations,
            'is_test': self.is_test,
            'use_mkldnn': self.use_mkldnn,
            'mkldnn_data_type': self.mkldnn_data_type,
            'force_fp32_output': self.force_fp32_output,
            'data_format': self.data_format,
            'Scale_in': self.scale_in,
            'Scale_out': self.scale_out,
            'Scale_weights': self.scale_weights,
            'fuse_activation': self.fuse_activation,
            'fuse_alpha': self.fuse_alpha,
            'fuse_beta': self.fuse_beta,
        }
        if self.output_size is not None:
            self.attrs['output_size'] = self.output_size

        if len(self.output_padding) > 0:
            self.attrs['output_padding'] = self.output_padding

        # This implementation of convolution quantization is based on OneDNN documentation
        # https://oneapi-src.github.io/oneDNN/dev_guide_int8_computations.html#doxid-dev-guide-int8-computations-1dg-i8-comp-s11
        inner_scale = 1.0 if self.fuse_activation != "" else self.scale_out
        activation_scale = self.scale_out if self.fuse_activation != "" else 1.0
        scale_output_shift = inner_scale / (
            self.scale_in * self.scale_weights[0]
        )
        filter = np.random.random(self.filter_size).astype(self.weighttype)

        # When the Intel AVX2 or Intel AVX512 Instruction Set is used
        # the reorder additionally scales the weights by 0.5
        # to overcome the potential overflow issue. If the processor supports VNNI instructions,
        # modification of the weights is not necessary.
        avx_scale = (
            0.5 if not core.supports_vnni() and self.srctype == np.int8 else 1.0
        )
        filter_int = np.round(
            filter  # * self.scale_weights[0] * avx_scale
        ).astype(np.int32)
        scale_output_shift = scale_output_shift / avx_scale

        def conv2dtranspose_forward_helper(input_):
            return conv2dtranspose_forward_naive(
                input_.astype(np.int32), filter_int, self.attrs
            ).astype(
                np.float32
            )  # * scale_output_shift)

        if self.srctype == np.int8:
            init_low, init_high = (-5, 5)
            input = np.random.randint(
                init_low, init_high, self.input_size
            ).astype(self.srctype)
            input_shift = (np.ones(self.input_size) * 128).astype(np.uint8)

            output1 = conv2dtranspose_forward_helper(
                np.round(input + input_shift).astype(np.int32)
            )
            output2 = conv2dtranspose_forward_helper(
                np.round(input_shift).astype(np.int32)
            )
            output = output1 - output2
        else:
            init_low, init_high = (0, 10)
            input = np.random.randint(
                init_low, init_high, self.input_size
            ).astype(self.srctype)
            output = conv2dtranspose_forward_helper(input)

        # self.inputs = {
        #     'Input': input.view(self.input_type),
        #     'Filter': OpTest.np_dtype_to_fluid_dtype(filter),
        # }

        # if self.fuse_bias and self.bias_size is not None:
        #     bias = np.random.random(self.bias_size).astype(np.int8)
        #     output = conv2d_bias_naive(output, bias)
        #     output = output.astype(np.int8)
        #     self.attrs['fuse_bias'] = self.fuse_bias
        #     self.inputs['Bias'] = OpTest.np_dtype_to_fluid_dtype(bias)

        if self.fuse_activation == "relu":
            output = activation_scale * np.maximum(output, 0)

        output = np.round(output).astype(self.dsttype)
        self.inputs = {
            'Input': OpTest.np_dtype_to_fluid_dtype(input.astype(self.srctype)),
            'Filter': OpTest.np_dtype_to_fluid_dtype(filter),
        }
        self.outputs = {'Output': output}


# class TestMKLDNNFuseBias(TestConv2DTransposeINT8MKLDNNOp):
#     def init_test_case(self):
#         super().init_test_case()
#         self.pad = [1, 1]
#         self.fuse_bias = True
#         self.bias_size = [6]


class TestMKLDNNWithPad(TestConv2DTransposeINT8MKLDNNOp):
    def init_test_case(self):
        super().init_test_case()
        self.pad = [1, 1]
        self.input_size = [2, 3, 10, 10]


class TestMKLDNNWithStride(TestConv2DTransposeINT8MKLDNNOp):
    def init_test_case(self):
        super().init_test_case()
        self.pad = [1, 1]
        self.stride = [2, 2]
        self.input_size = [2, 3, 6, 6]  # NCHW


class TestMKLDNNWithAsymPad(TestConv2DTransposeINT8MKLDNNOp):
    def init_test_case(self):
        super().init_test_case()
        self.pad = [0, 0, 1, 2]
        self.padding_algorithm = "EXPLICIT"


class TestMKLDNNWithSamePad(TestConv2DTransposeINT8MKLDNNOp):
    def init_test_case(self):
        super().init_test_case()
        self.pad = [0, 0]
        self.padding_algorithm = "SAME"


class TestMKLDNNWithValidPad(TestConv2DTransposeINT8MKLDNNOp):
    def init_test_case(self):
        super().init_test_case()
        self.pad = [1, 1]
        self.padding_algorithm = "VALID"


class TestMKLDNNWithValidPad_NHWC(TestMKLDNNWithValidPad):
    def init_test_case(self):
        super().init_test_case()
        self.data_format = 'NHWC'
        N, C, H, W = self.input_size
        self.input_size = [N, H, W, C]


class TestConv2DTransposeMKLDNNWithDilationsExplicitPad(
    TestConv2DTransposeINT8MKLDNNOp
):
    def init_test_case(self):
        super().init_test_case()
        self.stride = [2, 1]
        self.dilations = [1, 2]
        self.groups = 1
        self.input_size = [4, 3, 8, 7]  # NCHW
        f_c = self.input_size[1]
        self.filter_size = [f_c, 6, 4, 3]
        self.pad = [1, 3, 2, 1]
        self.padding_algorithm = "EXPLICIT"


if __name__ == '__main__':
    enable_static()
    unittest.main()
