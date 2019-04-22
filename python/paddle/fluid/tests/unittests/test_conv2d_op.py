#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import paddle.fluid.core as core
from op_test import OpTest


def conv2d_forward_naive(input, filter, group, conv_param):
    in_n, in_c, in_h, in_w = input.shape
    out_c, f_c, f_h, f_w = filter.shape
    assert f_c * group == in_c
    assert np.mod(out_c, group) == 0
    sub_out_c = out_c // group

    stride, pad, dilation = conv_param['stride'], conv_param['pad'], conv_param[
        'dilation']
    out_h = 1 + (in_h + 2 * pad[0] - (dilation[0] * (f_h - 1) + 1)) // stride[0]
    out_w = 1 + (in_w + 2 * pad[1] - (dilation[1] * (f_w - 1) + 1)) // stride[1]
    out = np.zeros((in_n, out_c, out_h, out_w))

    d_bolck_h = (dilation[0] * (f_h - 1) + 1)
    d_bolck_w = (dilation[1] * (f_w - 1) + 1)

    input_pad = np.pad(input, ((0, ), (0, ), (pad[0], ), (pad[1], )),
                       mode='constant',
                       constant_values=0)

    filter_dilation = np.zeros((out_c, f_c, d_bolck_h, d_bolck_w))
    filter_dilation[:, :, 0:d_bolck_h:dilation[0], 0:d_bolck_w:dilation[
        1]] = filter

    for i in range(out_h):
        for j in range(out_w):
            for g in range(group):
                input_pad_masked = \
                    input_pad[:, g * f_c:(g + 1) * f_c,
                    i * stride[0]:i * stride[0] + d_bolck_h,
                    j * stride[1]:j * stride[1] + d_bolck_w]

                f_sub = filter_dilation[g * sub_out_c:(g + 1) *
                                        sub_out_c, :, :, :]
                for k in range(sub_out_c):
                    out[:, g * sub_out_c + k, i, j] = \
                        np.sum(input_pad_masked * f_sub[k, :, :, :],
                               axis=(1, 2, 3))

    return out, in_n, out_h, out_w, out_c


class TestConv2dOp(OpTest):
    def setUp(self):
        self.op_type = "conv2d"
        self.use_cudnn = False
        self.exhaustive_search = False
        self.use_cuda = False
        self.use_mkldnn = False
        self.fuse_relu_before_depthwise_conv = False
        self.data_format = "AnyLayout"
        self.dtype = np.float32
        self.init_kernel_type()
        self.init_group()
        self.init_dilation()
        self.init_test_case()

        conv2d_param = {
            'stride': self.stride,
            'pad': self.pad,
            'dilation': self.dilations
        }

        input = np.random.random(self.input_size).astype(self.dtype)
        if not self.has_cuda():
            self.fuse_relu_before_depthwise_conv = False
        if self.fuse_relu_before_depthwise_conv:
            input = input - 0.5
            input -= (input < 0) * 0.1
            input += (input >= 0) * 0.1
            input2 = np.maximum(input, 0.0)
        else:
            input2 = input
        filter = np.random.random(self.filter_size).astype(self.dtype)
        output, _, _, _, _ = conv2d_forward_naive(input2, filter, self.groups,
                                                  conv2d_param)
        output = output.astype(self.dtype)

        self.inputs = {
            'Input': OpTest.np_dtype_to_fluid_dtype(input),
            'Filter': OpTest.np_dtype_to_fluid_dtype(filter)
        }
        self.attrs = {
            'strides': self.stride,
            'paddings': self.pad,
            'groups': self.groups,
            'dilations': self.dilations,
            'use_cudnn': self.use_cudnn,
            'use_mkldnn': self.use_mkldnn,
            'data_format': self.data_format,
            'fuse_relu_before_depthwise_conv':
            self.fuse_relu_before_depthwise_conv,
            'exhaustive_search': self.exhaustive_search
        }
        self.outputs = {'Output': output}

    def has_cuda(self):
        return core.is_compiled_with_cuda() and (self.use_cudnn or
                                                 self.use_cuda)

    def test_check_output(self):
        place = core.CUDAPlace(0) if self.has_cuda() else core.CPUPlace()
        self.check_output_with_place(place, atol=1e-5)

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        place = core.CUDAPlace(0) if self.has_cuda() else core.CPUPlace()
        self.check_grad_with_place(
            place, {'Input', 'Filter'}, 'Output', max_relative_error=0.02)

    def test_check_grad_no_filter(self):
        if self.dtype == np.float16:
            return
        place = core.CUDAPlace(0) if self.has_cuda() else core.CPUPlace()
        self.check_grad_with_place(
            place, ['Input'],
            'Output',
            max_relative_error=0.02,
            no_grad_set=set(['Filter']))

    def test_check_grad_no_input(self):
        if self.dtype == np.float16:
            return
        place = core.CUDAPlace(0) if self.has_cuda() else core.CPUPlace()
        self.check_grad_with_place(
            place, ['Filter'],
            'Output',
            max_relative_error=0.02,
            no_grad_set=set(['Input']))

    def init_test_case(self):
        self.pad = [0, 0]
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 3]

    def init_dilation(self):
        self.dilations = [1, 1]

    def init_group(self):
        self.groups = 1

    def init_kernel_type(self):
        pass


class TestWithPad(TestConv2dOp):
    def init_test_case(self):
        self.pad = [1, 1]
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 3]


class TestWithStride(TestConv2dOp):
    def init_test_case(self):
        self.pad = [1, 1]
        self.stride = [2, 2]
        self.input_size = [2, 3, 6, 6]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 3]


class TestWithGroup(TestConv2dOp):
    def init_group(self):
        self.groups = 3


class TestWith1x1(TestConv2dOp):
    def init_test_case(self):
        self.pad = [0, 0]
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 1, 1]

    def init_group(self):
        self.groups = 3


class TestWithDilation(TestConv2dOp):
    def init_test_case(self):
        self.pad = [0, 0]
        self.stride = [1, 1]
        self.input_size = [2, 3, 10, 10]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 3]

    def init_dilation(self):
        self.dilations = [2, 2]

    def init_group(self):
        self.groups = 3


class TestWithInput1x1Filter1x1(TestConv2dOp):
    def init_test_case(self):
        self.pad = [0, 0]
        self.stride = [1, 1]
        self.input_size = [2, 3, 1, 1]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 1, 1]

    def init_group(self):
        self.groups = 3


#----------------Conv2dCUDNN----------------


def create_test_cudnn_class(parent):
    @unittest.skipIf(not core.is_compiled_with_cuda(),
                     "core is not compiled with CUDA")
    class TestCUDNNCase(parent):
        def init_kernel_type(self):
            self.use_cudnn = True

    cls_name = "{0}_{1}".format(parent.__name__, "CUDNN")
    TestCUDNNCase.__name__ = cls_name
    globals()[cls_name] = TestCUDNNCase


create_test_cudnn_class(TestConv2dOp)
create_test_cudnn_class(TestWithPad)
create_test_cudnn_class(TestWithStride)
create_test_cudnn_class(TestWithGroup)
create_test_cudnn_class(TestWith1x1)
create_test_cudnn_class(TestWithInput1x1Filter1x1)

#----------------Conv2dCUDNN----------------


def create_test_cudnn_fp16_class(parent, grad_check=True):
    @unittest.skipIf(not core.is_compiled_with_cuda(),
                     "core is not compiled with CUDA")
    class TestConv2DCUDNNFp16(parent):
        def init_kernel_type(self):
            self.use_cudnn = True
            self.dtype = np.float16

        def test_check_output(self):
            if core.is_compiled_with_cuda():
                place = core.CUDAPlace(0)
                if core.is_float16_supported(place):
                    self.check_output_with_place(place, atol=2e-2)

        def test_check_grad_no_filter(self):
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place) and grad_check:
                self.check_grad_with_place(
                    place, ['Input'],
                    'Output',
                    max_relative_error=0.02,
                    no_grad_set=set(['Filter']))

        def test_check_grad_no_input(self):
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place) and grad_check:
                self.check_grad_with_place(
                    place, ['Filter'],
                    'Output',
                    max_relative_error=0.02,
                    no_grad_set=set(['Input']))

    cls_name = "{0}_{1}".format(parent.__name__, "CUDNNFp16")
    TestConv2DCUDNNFp16.__name__ = cls_name
    globals()[cls_name] = TestConv2DCUDNNFp16


create_test_cudnn_fp16_class(TestConv2dOp, grad_check=False)
create_test_cudnn_fp16_class(TestWithPad, grad_check=False)
create_test_cudnn_fp16_class(TestWithStride, grad_check=False)
create_test_cudnn_fp16_class(TestWithGroup, grad_check=False)
create_test_cudnn_fp16_class(TestWith1x1, grad_check=False)
create_test_cudnn_fp16_class(TestWithInput1x1Filter1x1, grad_check=False)

# -------TestDepthwiseConv


class TestDepthwiseConv(TestConv2dOp):
    def init_test_case(self):
        self.use_cuda = True
        self.pad = [1, 1]
        self.stride = [2, 2]
        self.input_size = [2, 3, 5, 5]  # NCHW
        self.groups = 3
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [3, f_c, 3, 3]
        self.op_type = "depthwise_conv2d"


class TestDepthwiseConv2(TestConv2dOp):
    def init_test_case(self):
        self.use_cuda = True
        self.pad = [1, 1]
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]  # NCHW
        self.groups = 3
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [3, f_c, 3, 3]
        self.op_type = "depthwise_conv2d"


class TestDepthwiseConv3(TestConv2dOp):
    def init_test_case(self):
        self.use_cuda = True
        self.pad = [1, 1]
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]  # NCHW
        self.groups = 3
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 3]
        self.op_type = "depthwise_conv2d"


class TestDepthwiseConvWithDilation(TestConv2dOp):
    def init_test_case(self):
        self.use_cuda = True
        self.pad = [1, 1]
        self.stride = [2, 2]
        self.input_size = [2, 3, 5, 5]  # NCHW
        self.groups = 3
        self.dilations = [2, 2]
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 3]
        self.op_type = "depthwise_conv2d"


class TestDepthwiseConvWithDilation2(TestConv2dOp):
    def init_test_case(self):
        self.use_cuda = True
        self.pad = [1, 1]
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]  # NCHW
        self.groups = 3
        self.dilations = [2, 2]
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 3]
        self.op_type = "depthwise_conv2d"


class TestDepthwiseConvandFuse(TestConv2dOp):
    def init_test_case(self):
        self.fuse_relu_before_depthwise_conv = True
        self.use_cuda = True
        self.pad = [1, 1]
        self.stride = [2, 2]
        self.input_size = [2, 3, 5, 5]  # NCHW
        self.groups = 3
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [3, f_c, 3, 3]
        self.op_type = "depthwise_conv2d"


class TestDepthwiseConv2andFuse(TestConv2dOp):
    def init_test_case(self):
        self.fuse_relu_before_depthwise_conv = True
        self.use_cuda = True
        self.pad = [1, 1]
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]  # NCHW
        self.groups = 3
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [3, f_c, 3, 3]
        self.op_type = "depthwise_conv2d"


class TestDepthwiseConv3andFuse(TestConv2dOp):
    def init_test_case(self):
        self.fuse_relu_before_depthwise_conv = True
        self.use_cuda = True
        self.pad = [1, 1]
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]  # NCHW
        self.groups = 3
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 3]
        self.op_type = "depthwise_conv2d"


class TestDepthwiseConvWithDilationandFuse(TestConv2dOp):
    def init_test_case(self):
        self.fuse_relu_before_depthwise_conv = True
        self.use_cuda = True
        self.pad = [1, 1]
        self.stride = [2, 2]
        self.input_size = [2, 3, 5, 5]  # NCHW
        self.groups = 3
        self.dilations = [2, 2]
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 3]
        self.op_type = "depthwise_conv2d"


class TestDepthwiseConvWithDilation2andFuse(TestConv2dOp):
    def init_test_case(self):
        self.fuse_relu_before_depthwise_conv = True
        self.use_cuda = True
        self.pad = [1, 1]
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]  # NCHW
        self.groups = 3
        self.dilations = [2, 2]
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 3]
        self.op_type = "depthwise_conv2d"


class TestCUDNNExhaustiveSearch(TestConv2dOp):
    def init_kernel_type(self):
        self.use_cudnn = True
        self.exhaustive_search = True


# Please Don't remove the following code.
# Currently, CI use cudnn V5.0 which not support dilation conv.
# class TestCUDNNWithDilation(TestWithDilation):
#     def init_op_type(self):
#         self.op_type = "conv_cudnn"

if __name__ == '__main__':
    unittest.main()
