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

import unittest
from itertools import product

import numpy as np

import paddle

paddle.enable_static()


def dmc_bilinear(data_im, height, width, h, w):
    h_low = int(np.floor(h))
    w_low = int(np.floor(w))
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


def dconv_im2col_gemm(input, offset, mask, filter, group, conv_param):
    in_n, in_c, in_h, in_w = input.shape
    out_c, f_c, f_h, f_w = filter.shape

    assert offset.shape == (in_n, 2 * f_h * f_w, in_h, in_w)
    assert mask.shape == (in_n, f_h * f_w, in_h, in_w)
    assert f_c * group == in_c
    assert np.mod(out_c, group) == 0

    stride, pad, dilation = (
        conv_param['stride'],
        conv_param['pad'],
        conv_param['dilation'],
    )
    out_h = 1 + (in_h + 2 * pad[0] - (dilation[0] * (f_h - 1) + 1)) // stride[0]
    out_w = 1 + (in_w + 2 * pad[1] - (dilation[1] * (f_w - 1) + 1)) // stride[1]
    assert out_h == in_h
    assert out_w == in_w

    col_buffer = np.zeros((in_n, in_c * f_h * f_w, in_h * in_w))
    for n, c, h, w, kh, kw in product(
        range(in_n),
        range(in_c),
        range(out_h),
        range(out_w),
        range(f_h),
        range(f_w),
    ):
        offset_h_table = offset[n, ::2, h, w].reshape(f_h, f_w)
        offset_w_table = offset[n, 1::2, h, w].reshape(f_h, f_w)
        mask_table = mask[n, :, h, w].reshape(f_h, f_w)
        offset_h = offset_h_table[kh, kw]
        offset_w = offset_w_table[kh, kw]
        val = 0
        im_h = h * stride[0] + kh * dilation[0] + offset_h - pad[0]
        im_w = w * stride[0] + kw * dilation[0] + offset_w - pad[1]
        if im_h > -1 and im_w > -1 and im_h < in_h and im_w < in_h:
            val = dmc_bilinear(input[n, c], in_h, in_w, im_h, im_w)
        val_out = val * mask_table[kh, kw]
        col_buffer[n, c * f_h * f_w + kh * f_w + kw, h * in_w + w] = val_out

    out = np.zeros((in_n, group, int(out_c // group), out_h * out_w))
    weight = filter.reshape(group, int(out_c // group), f_c * f_h * f_w)
    col_buffer = col_buffer.reshape(
        (in_n, group, int(in_c // group * f_h * f_w), in_h * in_w)
    )
    for n in range(in_n):
        for g in range(group):
            out[n, g] = np.matmul(weight[g], col_buffer[n, g])
    out = out.reshape(in_n, out_c, out_h, out_w)
    return out


class TestModulatedDeformableConvInvalidInput(unittest.TestCase):
    def test_error(self):
        def test_invalid_input():
            paddle.enable_static()
            input = [1, 3, 32, 32]
            offset = paddle.static.data(
                name='offset', shape=[None, 3, 32, 32], dtype='float32'
            )
            mask = paddle.static.data(
                name='mask', shape=[None, 3, 32, 32], dtype='float32'
            )
            loss = paddle.static.nn.common.deformable_conv(
                input, offset, mask, num_filters=4, filter_size=1
            )

        self.assertRaises(TypeError, test_invalid_input)

        def test_invalid_offset():
            paddle.enable_static()
            input = paddle.static.data(
                name='input', shape=[None, 3, 32, 32], dtype='int32'
            )
            offset = paddle.static.data(
                name='offset', shape=[None, 3, 32, 32], dtype='float32'
            )
            mask = paddle.static.data(
                name='mask', shape=[None, 3, 32, 32], dtype='float32'
            )
            loss = paddle.static.nn.common.deformable_conv(
                input, offset, mask, num_filters=4, filter_size=1
            )

        self.assertRaises(TypeError, test_invalid_offset)

        def test_invalid_filter():
            paddle.enable_static()
            input = paddle.static.data(
                name='input_filter', shape=[None, 3, 32, 32], dtype='float32'
            )
            offset = paddle.static.data(
                name='offset_filter', shape=[None, 3, 32, 32], dtype='float32'
            )
            mask = paddle.static.data(
                name='mask_filter', shape=[None, 3, 32, 32], dtype='float32'
            )
            loss = paddle.static.nn.common.deformable_conv(
                input, offset, mask, num_filters=4, filter_size=0
            )

        self.assertRaises(ValueError, test_invalid_filter)

        def test_invalid_groups():
            paddle.enable_static()
            input = paddle.static.data(
                name='input_groups', shape=[1, 1, 1, 1], dtype='float32'
            )
            offset = paddle.static.data(
                name='offset_groups', shape=[1, 1], dtype='float32'
            )
            mask = paddle.static.data(
                name='mask_groups', shape=[1], dtype='float32'
            )
            paddle.static.nn.deform_conv2d(
                input, offset, mask, 1, 1, padding=1, groups=0
            )

        self.assertRaises(ValueError, test_invalid_groups)


if __name__ == '__main__':
    unittest.main()
