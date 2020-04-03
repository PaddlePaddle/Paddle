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
from op_test import OpTest
import paddle.fluid.core as core
import paddle.fluid as fluid


def cubic_1(x, a):
    return ((a + 2) * x - (a + 3)) * x * x + 1


def cubic_2(x, a):
    return ((a * x - 5 * a) * x + 8 * a) * x - 4 * a


def cubic_interp1d(x0, x1, x2, x3, t):
    param = [0, 0, 0, 0]
    a = -0.75
    x_1 = t
    x_2 = 1.0 - t
    param[0] = cubic_2(x_1 + 1.0, a)
    param[1] = cubic_1(x_1, a)
    param[2] = cubic_1(x_2, a)
    param[3] = cubic_2(x_2 + 1.0, a)
    return x0 * param[0] + x1 * param[1] + x2 * param[2] + x3 * param[3]


def value_bound(input, w, h, x, y):
    access_x = int(max(min(x, w - 1), 0))
    access_y = int(max(min(y, h - 1), 0))
    return input[:, :, access_y, access_x]


def bicubic_interp_np(input,
                      out_h,
                      out_w,
                      out_size=None,
                      actual_shape=None,
                      align_corners=True,
                      data_layout='kNCHW'):
    """trilinear interpolation implement in shape [N, C, H, W]"""
    if data_layout == "NHWC":
        input = np.transpose(input, (0, 3, 1, 2))  # NHWC => NCHW
    if out_size is not None:
        out_h = out_size[0]
        out_w = out_size[1]
    if actual_shape is not None:
        out_h = actual_shape[0]
        out_w = actual_shape[1]
    batch_size, channel, in_h, in_w = input.shape

    ratio_h = ratio_w = 0.0
    if out_h > 1:
        if (align_corners):
            ratio_h = (in_h - 1.0) / (out_h - 1.0)
        else:
            ratio_h = 1.0 * in_h / out_h

    if out_w > 1:
        if (align_corners):
            ratio_w = (in_w - 1.0) / (out_w - 1.0)
        else:
            ratio_w = 1.0 * in_w / out_w

    out = np.zeros((batch_size, channel, out_h, out_w))

    for k in range(out_h):
        if (align_corners):
            h = ratio_h * k
        else:
            h = ratio_h * (k + 0.5) - 0.5
        input_y = int(h)
        y_t = h - input_y
        for l in range(out_w):
            if (align_corners):
                w = ratio_w * l
            else:
                w = ratio_w * (l + 0.5) - 0.5
            input_x = int(w)
            x_t = w - input_x
            for i in range(batch_size):
                for j in range(channel):
                    coefficients = [0, 0, 0, 0]
                    for ii in range(4):
                        access_x_0 = int(max(min(input_x - 1, in_w - 1), 0))
                        access_x_1 = int(max(min(input_x + 0, in_w - 1), 0))
                        access_x_2 = int(max(min(input_x + 1, in_w - 1), 0))
                        access_x_3 = int(max(min(input_x + 2, in_w - 1), 0))
                        access_y = int(max(min(input_y - 1 + ii, in_h - 1), 0))

                        coefficients[ii] = cubic_interp1d(
                            input[i, j, access_y, access_x_0],
                            input[i, j, access_y, access_x_1],
                            input[i, j, access_y, access_x_2],
                            input[i, j, access_y, access_x_3], x_t)
                    out[i, j, k, l] = cubic_interp1d(
                        coefficients[0], coefficients[1], coefficients[2],
                        coefficients[3], y_t)
    if data_layout == "NHWC":
        out = np.transpose(out, (0, 2, 3, 1))  # NCHW => NHWC
    return out.astype(input.dtype)


class TestBicubicInterpOp(OpTest):
    def setUp(self):
        self.out_size = None
        self.actual_shape = None
        self.data_layout = 'NCHW'
        self.init_test_case()
        self.op_type = "bicubic_interp"
        input_np = np.random.random(self.input_shape).astype("float64")

        if self.data_layout == "NCHW":
            in_h = self.input_shape[2]
            in_w = self.input_shape[3]
        else:
            in_h = self.input_shape[1]
            in_w = self.input_shape[2]

        if self.scale > 0:
            out_h = int(in_h * self.scale)
            out_w = int(in_w * self.scale)
        else:
            out_h = self.out_h
            out_w = self.out_w

        output_np = bicubic_interp_np(input_np, out_h, out_w, self.out_size,
                                      self.actual_shape, self.align_corners,
                                      self.data_layout)
        self.inputs = {'X': input_np}
        if self.out_size is not None:
            self.inputs['OutSize'] = self.out_size
        if self.actual_shape is not None:
            self.inputs['OutSize'] = self.actual_shape

        self.attrs = {
            'out_h': self.out_h,
            'out_w': self.out_w,
            'scale': self.scale,
            'interp_method': self.interp_method,
            'align_corners': self.align_corners,
            'data_layout': self.data_layout
        }
        self.outputs = {'Out': output_np}

    def test_check_output(self):
        self.check_output()
        self.check_output_with_place(place=core.CUDAPlace(0))

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', in_place=True)

    def init_test_case(self):
        self.interp_method = 'bicubic'
        self.input_shape = [2, 3, 5, 5]
        self.out_h = 2
        self.out_w = 2
        self.scale = 0.
        self.out_size = np.array([3, 3]).astype("int32")
        self.align_corners = True


class TestBicubicInterpCase1(TestBicubicInterpOp):
    def init_test_case(self):
        self.interp_method = 'bicubic'
        self.input_shape = [4, 1, 7, 8]
        self.out_h = 1
        self.out_w = 1
        self.scale = 0.
        self.align_corners = True


class TestBicubicInterpCase2(TestBicubicInterpOp):
    def init_test_case(self):
        self.interp_method = 'bicubic'
        self.input_shape = [3, 3, 9, 6]
        self.out_h = 10
        self.out_w = 8
        self.scale = 0.
        self.align_corners = True


class TestBicubicInterpCase3(TestBicubicInterpOp):
    def init_test_case(self):
        self.interp_method = 'bicubic'
        self.input_shape = [1, 1, 32, 64]
        self.out_h = 64
        self.out_w = 32
        self.scale = 0.
        self.align_corners = False


class TestBicubicInterpCase4(TestBicubicInterpOp):
    def init_test_case(self):
        self.interp_method = 'bicubic'
        self.input_shape = [4, 1, 7, 8]
        self.out_h = 1
        self.out_w = 1
        self.scale = 0.
        self.out_size = np.array([2, 2]).astype("int32")
        self.align_corners = True


class TestBicubicInterpCase5(TestBicubicInterpOp):
    def init_test_case(self):
        self.interp_method = 'bicubic'
        self.input_shape = [3, 3, 9, 6]
        self.out_h = 11
        self.out_w = 11
        self.scale = 0.
        self.out_size = np.array([6, 4]).astype("int32")
        self.align_corners = False


class TestBicubicInterpCase6(TestBicubicInterpOp):
    def init_test_case(self):
        self.interp_method = 'bicubic'
        self.input_shape = [1, 1, 32, 64]
        self.out_h = 64
        self.out_w = 32
        self.scale = 0.
        self.out_size = np.array([64, 32]).astype("int32")
        self.align_corners = False


class TestBicubicInterpSame(TestBicubicInterpOp):
    def init_test_case(self):
        self.interp_method = 'bicubic'
        self.input_shape = [2, 3, 32, 64]
        self.out_h = 32
        self.out_w = 64
        self.scale = 0.
        self.align_corners = True


class TestBicubicInterpDataLayout(TestBicubicInterpOp):
    def init_test_case(self):
        self.interp_method = 'bicubic'
        self.input_shape = [2, 5, 5, 3]
        self.out_h = 2
        self.out_w = 2
        self.scale = 0.
        self.out_size = np.array([3, 3]).astype("int32")
        self.align_corners = True
        self.data_layout = "NHWC"


if __name__ == "__main__":
    unittest.main()
