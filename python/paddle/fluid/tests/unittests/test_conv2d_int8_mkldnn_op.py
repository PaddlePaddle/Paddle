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
    out_tmp = np.zeros((in_n, out_h, out_w, out_c))
    for n in range(in_n):
        for i in range(out_h):
            for j in range(out_w):
                for m in range(out_c):
                    out_tmp[n, i, j, m] = out[n, m, i, j]

    out = out_tmp.reshape(in_n, out_c, out_h, out_w)
    return out


class TestConv2dOp(OpTest):
    def setUp(self):
        self.op_type = "conv2d"
        self.use_cudnn = False
        self.exhaustive_search = False
        self.use_cuda = False
        self.use_mkldnn = False
        self.data_format = "AnyLayout"
        self.weighttype = np.float32
        self.use_mkldnn = True
        self.init_group()
        self.init_dilation()
        self.init_test_case()
        self.init_kernel_type()

        conv2d_param = {
            'stride': self.stride,
            'pad': self.pad,
            'dilation': self.dilations
        }

        filter = np.random.random(self.filter_size).astype(self.weighttype)

        if self.srctype == np.uint8:
            input = np.random.randint(0, 10,
                                      self.input_size).astype(self.srctype)
        else:
            input = np.random.randint(-5, 5,
                                      self.input_size).astype(self.srctype)
            input_shift = (np.ones(self.input_size) * 128).astype(np.uint8)

        if self.srctype == np.int8:
            output1 = conv2d_forward_naive(
                np.round((input.astype(np.int32) + input_shift) *
                         self.scale_in).astype(np.int32),
                np.round(filter * self.scale_weights[0] * 0.5).astype(np.int32),
                self.groups, conv2d_param).astype(np.float32) * (
                    self.scale_out / (self.scale_in * self.scale_weights[0] *
                                      0.5))  #).astype(self.dsttype)
            output2 = conv2d_forward_naive(
                np.round((input_shift) * self.scale_in).astype(np.int32),
                np.round(filter * self.scale_weights[0] * 0.5).astype(np.int32),
                self.groups, conv2d_param).astype(np.float32) * (
                    self.scale_out / (self.scale_in * self.scale_weights[0] *
                                      0.5))  #).astype(self.dsttype)
            output = np.round(output1 - output2).astype(self.dsttype)
        else:
            output1 = conv2d_forward_naive(
                input.astype(np.int32),
                np.round(filter * self.scale_weights[0]).astype(np.int32),
                self.groups, conv2d_param).astype(np.float32)
            output = np.round(output1 * (self.scale_out / (
                self.scale_in * self.scale_weights[0]))).astype(self.dsttype)

        self.inputs = {
            'Input':
            OpTest.np_dtype_to_fluid_dtype(input.astype(self.srctype)),
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
            'exhaustive_search': self.exhaustive_search,
            'Scale_in': self.scale_in,
            'Scale_out': self.scale_out,
            'Scale_weights': self.scale_weights,
        }
        self.outputs = {'Output': output}

    def test_check_output(self):
        self.check_output_with_place(core.CPUPlace(), atol=0)

    def init_test_case(self):
        self.pad = [0, 0]
        self.stride = [1, 1]
        self.input_size = [22, 3, 5, 5]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [1, f_c, 3, 3]
        self.scale_in = 1.0
        self.scale_out = 0.5
        self.scale_weights = [10.0]

    def init_dilation(self):
        self.dilations = [1, 1]

    def init_group(self):
        self.groups = 1

    def init_kernel_type(self):
        self.srctype = np.uint8
        self.dsttype = np.float32


class TestMKLDNN(TestConv2dOp):
    def init_test_case(self):
        self.pad = [0, 0]
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 3]
        self.scale_in = 1.0
        self.scale_out = 0.5
        self.scale_weights = [10.0]


class TestWithPad(TestConv2dOp):
    def init_test_case(self):
        self.pad = [1, 1]
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 3]
        self.scale_in = 1.0
        self.scale_out = 1.5
        self.scale_weights = [10.0]


class TestWithStride(TestConv2dOp):
    def init_test_case(self):
        self.pad = [1, 1]
        self.stride = [2, 2]
        self.input_size = [2, 3, 6, 6]
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 3]
        self.scale_in = 1.0
        self.scale_out = 0.8
        self.scale_weights = [10.0]


class TestWithGroup(TestConv2dOp):
    def init_test_case(self):
        self.pad = [0, 0]
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 3]
        self.scale_in = 1.0
        self.scale_out = 0.5
        self.scale_weights = [10.0]

    def init_group(self):
        self.groups = 3


class TestWith1x1(TestConv2dOp):
    def init_test_case(self):
        self.pad = [0, 0]
        self.stride = [1, 1]
        self.input_size = [1, 3, 5, 5]
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 1, 1]
        self.scale_in = 1.0
        self.scale_out = 0.5
        self.scale_weights = [12.0]


class TestWithInput1x1Filter1x1(TestConv2dOp):
    def init_test_case(self):
        self.pad = [0, 0]
        self.stride = [1, 1]
        self.input_size = [2, 3, 1, 1]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 1, 1]
        self.scale_in = 1.0
        self.scale_out = 0.5
        self.scale_weights = [10.0]

    def init_group(self):
        self.groups = 3


class TestInt8Input(TestConv2dOp):
    def init_kernel_type(self):
        self.srctype = np.int8
        self.dsttype = np.int8


class TestMKLDNNWithPad(TestWithPad):
    def init_kernel_type(self):
        self.srctype = np.int8
        self.dsttype = np.int8


class TestMKLDNNWithStride(TestWithStride):
    def init_kernel_type(self):
        self.srctype = np.int8
        self.dsttype = np.int8


class TestMKLDNNWithGroup(TestWithGroup):
    def init_kernel_type(self):
        self.srctype = np.int8
        self.dsttype = np.int8


class TestMKLDNNWith1x1(TestWith1x1):
    def init_kernel_type(self):
        self.srctype = np.int8
        self.dsttype = np.int8


class TestMKLDNNWithInput1x1Filter1x1(TestWithInput1x1Filter1x1):
    def init_kernel_type(self):
        self.srctype = np.int8
        self.dsttype = np.int8


if __name__ == '__main__':
    unittest.main()
