#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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


def dmc_bilinear(data_im, height, width, h, w):
    h_low = np.floor(h)
    w_low = np.floor(w)
    h_high = h_low + 1
    w_high = w_low + 1

    lh = h - h_low
    lw = w - w_low
    hh = 1 - lh
    hw = 1 - lw

    v1 = 0
    if h_low >= 0 and w_low >= 0:
        v1 = data_im[h_low, w_low]
    v2 = 0
    if h_low >= 0 and w_high <= width - 1:
        v2 = data_im[h_low, w_high]
    v3 = 0
    if h_high <= height - 1 and w_low >= 0:
        v3 = data_im[h_high, w_low]
    v4 = 0
    if h_high <= height - 1 and w_high <= width - 1:
        v4 = data_im[h_high, w_high]

    w1, w2, w3, w4 = hh * hw, hh * lw, lh * hw, lh * lw
    val = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4
    return val


def dconv(input, offset, mask, filter):
    in_n, in_c, in_h, in_w = input.shape
    out_c, f_c, f_h, f_w = filter.shape

    assert offset.shape == (in_n, 2 * f_h * f_w, in_h, in_w)
    assert mask.shape == (in_n, f_h * f_w, in_h, in_w)
    assert f_c == in_c

    out_h = in_h
    out_w = in_w
    out = np.zeros((in_n, out_c, out_h, out_w))

    input_pad = np.pad(input, ((0, ), (0, ), ((f_h - 1) // 2, ), (
        (f_w - 1) // 2, )),
                       mode='constant',
                       constant_values=0)

    for n in range(in_n):
        for oc in range(oc):
            for i in range(out_h):
                for j in range(out_w):
                    offset_h = offset[n, :f_h * f_w, i, j].reshape(f_h, f_w)
                    offset_w = offset[n, f_h * f_w:, i, j].reshape(f_h, f_w)
                    mask_val = mask[n, :, i, j].reshape(f_h, f_w)
                    sub_conv_out = 0
                    for c in range(in_c):
                        for index_w in range(f_w):
                            for index_h in range(f_h):
                                # p0 + pn + /delta p
                                im_h = i + 1 + index_h - 1 + offset_h[index_h,
                                                                      index_w]
                                im_w = j + 1 + index_w - 1 + offset_w[index_h,
                                                                      index_w]
                                val = 0
                                data_im = input_pad[n, c]
                                if im_h > -1 and im_w > -1 and im_h < in_h and im_w < in_w:
                                    val = dmc_bilinear(data_im, im_h, im_w,
                                                       im_h, im_w)
                                val *= val * mask[index_h, index_w]
                                sub_conv_out += val * filter[oc, c, index_h,
                                                             index_w]
                    out[n, oc, i, j] = sub_conv_out
    return out


class TestConv2dOp(OpTest):
    def setUp(self):
        self.op_type = "modulated_deformable_conv"
        self.use_cuda = True
        self.dtype = np.float32
        self.init_group()
        self.init_dilation()
        self.init_test_case()

        conv2d_param = {
            'stride': self.stride,
            'pad': self.pad,
            'dilation': self.dilations
        }

        input = np.random.random(self.input_size).astype(self.dtype)
        offset = np.random.random(self.offset_size).astype(self.dtype)
        mask = np.random.random(self.mask_size).astype(self.dtype)
        filter = np.random.random(self.filter_size).astype(self.dtype)

        output = dconv(input, offset, mask, filter)
        output = output.astype(self.dtype)

        self.inputs = {
            'Input': OpTest.np_dtype_to_fluid_dtype(input),
            'Offset': OpTest.np_dtype_to_fluid_dtype(offset),
            'Mask': OpTest.np_dtype_to_fluid_dtype(mask),
            'Filter': OpTest.np_dtype_to_fluid_dtype(filter)
        }
        self.attrs = {
            'strides': self.stride,
            'paddings': self.pad,
            'groups': self.groups,
            'deformable_groups': self.deformable_groups,
            'im2col_step': self.im2col_step,
            'dilations': self.dilations,
        }
        self.outputs = {'Output': output}

    def test_check_output(self):
        place = core.CUDAPlace(0)
        self.check_output_with_place(place, atol=1e-5)

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        place = core.CUDAPlace(0)
        self.check_grad_with_place(
            place, {'Input', 'Offset', 'Mask', 'Filter'},
            'Output',
            max_relative_error=0.02)

    def init_test_case(self):
        self.pad = [1, 1]
        self.stride = [1, 1]
        self.dilations = [1, 1]
        self.input_size = [2, 3, 5, 5]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 3]

    def init_dilation(self):
        self.dilations = [1, 1]

    def init_group(self):
        self.groups = 1


class TestWithPad(TestConv2dOp):
    def init_test_case(self):
        self.pad = [1, 1]
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 3]
        self.offset_size = [2, 18, 5, 5]
        self.mask_size = [2, 9, 5, 5]


class TestWithStride(TestConv2dOp):
    def init_test_case(self):
        self.pad = [1, 1]
        self.stride = [2, 2]
        self.input_size = [2, 3, 6, 6]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 3]
        self.offset_size = [2, 18, 6, 6]
        self.mask_size = [2, 9, 6, 6]


class TestWithGroup(TestConv2dOp):
    def init_group(self):
        self.groups = 1


# class TestWith1x1(TestConv2dOp):
#     def init_test_case(self):
#         self.pad = [1, 1]
#         self.stride = [1, 1]
#         self.input_size = [2, 3, 5, 5]  # NCHW
#         assert np.mod(self.input_size[1], self.groups) == 0
#         f_c = self.input_size[1] // self.groups
#         self.filter_size = [6, f_c, 1, 1]

#     def init_group(self):
#         self.groups = 3

# class TestWithDilation(TestConv2dOp):
#     def init_test_case(self):
#         self.pad = [0, 0]
#         self.stride = [1, 1]
#         self.input_size = [2, 3, 10, 10]  # NCHW
#         assert np.mod(self.input_size[1], self.groups) == 0
#         f_c = self.input_size[1] // self.groups
#         self.filter_size = [6, f_c, 3, 3]

#     def init_dilation(self):
#         self.dilations = [2, 2]

#     def init_group(self):
#         self.groups = 3

# class TestWithInput1x1Filter1x1(TestConv2dOp):
#     def init_test_case(self):
#         self.pad = [0, 0]
#         self.stride = [1, 1]
#         self.input_size = [2, 3, 1, 1]  # NCHW
#         assert np.mod(self.input_size[1], self.groups) == 0
#         f_c = self.input_size[1] // self.groups
#         self.filter_size = [6, f_c, 1, 1]

#     def init_group(self):
#         self.groups = 3

if __name__ == '__main__':
    unittest.main()
