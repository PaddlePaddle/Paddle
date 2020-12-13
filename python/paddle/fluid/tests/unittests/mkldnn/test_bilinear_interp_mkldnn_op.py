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
import math
import unittest
import numpy as np
from paddle.fluid.tests.unittests.op_test import OpTest
import paddle.fluid.core as core
import paddle.fluid as fluid
from paddle.fluid.tests.unittests.op_test import skip_check_grad_ci

# def linear_map(y, y_max, x_max):
#     return (y + 0.5)*x_max/y_max - 0.5

# def left(y, y_max, x_max):
#     return max(int(math.floor(linear_map(y, y_max, x_max))), 0)

# def right(y, y_max, x_max):
#     return min(int(math.ceil(linear_map(y, y_max, x_max)), x_max - 1)

# def linear_weight(y, y_max, x_max):
#     return linear_map(y, y_max, x_max) - left(y, y_max, x_max)


def bilinear_interp_mkldnn_np(input,
                              out_h,
                              out_w,
                              out_size=None,
                              actual_shape=None,
                              data_layout='NCHW'):
    """bilinear interpolation implement in shape [N, C, H, W]"""
    if data_layout == "NHWC":
        input = np.transpose(input, (0, 3, 1, 2))  # NHWC => NCHW
    if out_size is not None:
        out_h = out_size[0]
        out_w = out_size[1]
    if actual_shape is not None:
        out_h = actual_shape[0]
        out_w = actual_shape[1]
    batch_size, channel, in_h, in_w = input.shape

    out = np.zeros((batch_size, channel, out_h, out_w))

    for oh in range(out_h):
        h0 = int(math.floor((oh + 0.5) * in_h / out_h - 0.5))
        h1 = int(math.ceil((oh + 0.5) * in_h / out_h - 0.5))
        h0 = max(h0, 0)
        # h1 = max(h1, 0)
        Wh = (oh + 0.5) * in_h / out_h - 0.5 - h0
        for ow in range(out_w):
            w0 = int(math.floor((ow + 0.5) * in_w / out_w - 0.5))
            w1 = int(math.ceil((ow + 0.5) * in_w / out_w - 0.5))
            w0 = max(w0, 0)
            # w1 = max(w1, 0)
            Ww = (ow + 0.5) * in_w / out_w - 0.5 - w0
            # h0 = min(h0, in_h - 1)
            h1 = min(h1, in_h - 1)
            # w0 = min(w0, in_w - 1)
            w1 = min(w1, in_w - 1)
            input_h0_w0 = input[:, :, h0, w0]
            input_h1_w0 = input[:, :, h1, w0]
            input_h0_w1 = input[:, :, h0, w1]
            input_h1_w1 = input[:, :, h1, w1]
            out[:, :, oh, ow] = input_h0_w0 * (1 - Wh) * (
                1 - Ww) + input_h1_w0 * Wh * (1 - Ww) + input_h0_w1 * (
                    1 - Wh) * Ww + input_h1_w1 * Wh * Ww

    if data_layout == "NHWC":
        out = np.transpose(out, (0, 2, 3, 1))  # NCHW => NHWC

    return out.astype(input.dtype)


@skip_check_grad_ci(reason="Haven not implement interpolate grad kernel.")
class TestBilinearInterpMKLDNNOp(OpTest):
    def setUp(self):
        self.out_size = None
        self.actual_shape = None
        self.data_layout = 'NCHW'
        self.init_test_case()
        self.op_type = "bilinear_interp"
        input_np = np.random.random(self.input_shape).astype("float32")

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

        output_np = bilinear_interp_mkldnn_np(input_np, out_h, out_w,
                                              self.out_size, self.actual_shape,
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
            'data_layout': self.data_layout,
            'use_mkldnn': self.use_mkldnn
        }
        self.outputs = {'Out': output_np}

    def test_check_output(self):
        print("Testing mkldnn bilinear kernel")
        self.check_output(check_dygraph=False)

    def init_test_case(self):
        # self.interp_method = 'bilinear'
        # self.input_shape = [2, 1, 2, 2]
        # self.out_h = 2
        # self.out_w = 2
        # self.scale = 0.
        # self.out_size = np.array([3, 3]).astype("int32")
        # self.use_mkldnn = True

        self.interp_method = 'bilinear'
        self.input_shape = [2, 5, 5, 3]
        self.out_h = 2
        self.out_w = 2
        self.scale = 0.
        self.out_size = np.array([3, 3]).astype("int32")
        self.data_layout = "NHWC"
        self.use_mkldnn = True


class TestBilinearInterpCase1(TestBilinearInterpMKLDNNOp):
    def init_test_case(self):
        self.interp_method = 'bilinear'
        self.input_shape = [4, 1, 7, 8]
        self.out_h = 1
        self.out_w = 1
        self.scale = 0.
        self.use_mkldnn = True


class TestBilinearInterpCase2(TestBilinearInterpMKLDNNOp):
    def init_test_case(self):
        self.interp_method = 'bilinear'
        self.input_shape = [1, 1, 9, 6]
        self.out_h = 12
        self.out_w = 12
        self.scale = 0.
        self.use_mkldnn = True
        self.data_layout = 'NCHW'


class TestBilinearInterpCase3(TestBilinearInterpMKLDNNOp):
    def init_test_case(self):
        self.interp_method = 'bilinear'
        self.input_shape = [1, 1, 32, 64]
        self.out_h = 64
        self.out_w = 32
        self.scale = 0.
        self.use_mkldnn = True


class TestBilinearInterpCase4(TestBilinearInterpMKLDNNOp):
    def init_test_case(self):
        self.interp_method = 'bilinear'
        self.input_shape = [4, 1, 7, 8]
        self.out_h = 1
        self.out_w = 1
        self.scale = 0.
        self.out_size = np.array([2, 2]).astype("int32")
        self.use_mkldnn = True


class TestBilinearInterpCase5(TestBilinearInterpMKLDNNOp):
    def init_test_case(self):
        self.interp_method = 'bilinear'
        self.input_shape = [3, 3, 9, 6]
        self.out_h = 12
        self.out_w = 12
        self.scale = 0.
        self.out_size = np.array([11, 11]).astype("int32")
        self.use_mkldnn = True


class TestBilinearInterpCase6(TestBilinearInterpMKLDNNOp):
    def init_test_case(self):
        self.interp_method = 'bilinear'
        self.input_shape = [2, 3, 32, 64]
        self.out_h = 64
        self.out_w = 32
        self.scale = 0.
        self.out_size = np.array([65, 33]).astype("int32")
        self.use_mkldnn = True


class TestBilinearInterpSame(TestBilinearInterpMKLDNNOp):
    def init_test_case(self):
        self.interp_method = 'bilinear'
        self.input_shape = [2, 3, 32, 64]
        self.out_h = 32
        self.out_w = 64
        self.scale = 0.
        self.use_mkldnn = True


class TestBilinearInterpActualShape(TestBilinearInterpMKLDNNOp):
    def init_test_case(self):
        self.interp_method = 'bilinear'
        self.input_shape = [3, 2, 32, 16]
        self.out_h = 64
        self.out_w = 32
        self.scale = 0.
        self.out_size = np.array([66, 40]).astype("int32")
        self.use_mkldnn = True


# class TestBilinearInterpDataLayout(TestBilinearInterpMKLDNNOp):
#     def init_test_case(self):
#         self.interp_method = 'bilinear'
#         self.input_shape = [2, 5, 5, 3]
#         self.out_h = 2
#         self.out_w = 2
#         self.scale = 0.
#         self.out_size = np.array([3, 3]).astype("int32")
#         self.data_layout = "NHWC"
#         self.use_mkldnn = True

if __name__ == "__main__":
    unittest.main()
