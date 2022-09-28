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

import unittest
import numpy as np
import math
import paddle
import paddle.fluid.core as core
import paddle.fluid as fluid
from paddle.fluid.tests.unittests.op_test import OpTest
from paddle.fluid.tests.unittests.op_test import skip_check_grad_ci


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
        h1 = min(h1, in_h - 1)
        Wh = (oh + 0.5) * in_h / out_h - 0.5 - h0
        for ow in range(out_w):
            w0 = int(math.floor((ow + 0.5) * in_w / out_w - 0.5))
            w1 = int(math.ceil((ow + 0.5) * in_w / out_w - 0.5))
            w0 = max(w0, 0)
            w1 = min(w1, in_w - 1)
            Ww = (ow + 0.5) * in_w / out_w - 0.5 - w0
            input_h0_w0 = input[:, :, h0, w0]
            input_h1_w0 = input[:, :, h1, w0]
            input_h0_w1 = input[:, :, h0, w1]
            input_h1_w1 = input[:, :, h1, w1]
            out[:, :, oh,
                ow] = input_h0_w0 * (1 - Wh) * (1 - Ww) + input_h1_w0 * Wh * (
                    1 - Ww) + input_h0_w1 * (1 -
                                             Wh) * Ww + input_h1_w1 * Wh * Ww

    if data_layout == "NHWC":
        out = np.transpose(out, (0, 2, 3, 1))  # NCHW => NHWC

    return out.astype(input.dtype)


@skip_check_grad_ci(reason="Haven not implement interpolate grad kernel.")
class TestBilinearInterpMKLDNNOp(OpTest):

    def init_test_case(self):
        pass

    def setUp(self):
        self.op_type = "bilinear_interp"
        self.interp_method = 'bilinear'
        self._cpu_only = True
        self.use_mkldnn = True
        self.input_shape = [1, 1, 2, 2]
        self.data_layout = 'NCHW'
        # priority: actual_shape > out_size > scale > out_h & out_w
        self.out_h = 1
        self.out_w = 1
        self.scale = 2.0
        self.out_size = None
        self.actual_shape = None

        self.init_test_case()

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
            'interp_method': self.interp_method,
            'out_h': self.out_h,
            'out_w': self.out_w,
            'scale': self.scale,
            'data_layout': self.data_layout,
            'use_mkldnn': self.use_mkldnn
        }
        self.outputs = {'Out': output_np}

    def test_check_output(self):
        self.check_output(check_dygraph=False)


class TestBilinearInterpOpMKLDNNNHWC(TestBilinearInterpMKLDNNOp):

    def init_test_case(self):
        self.input_shape = [3, 2, 32, 16]
        self.out_h = 27
        self.out_w = 49
        self.scale = 2.0
        self.data_layout = 'NHWC'


class TestBilinearNeighborInterpMKLDNNCase2(TestBilinearInterpMKLDNNOp):

    def init_test_case(self):
        self.input_shape = [3, 3, 9, 6]
        self.out_h = 12
        self.out_w = 12
        self.scale = 1.


class TestBilinearNeighborInterpDataLayout(TestBilinearInterpMKLDNNOp):

    def init_test_case(self):
        self.input_shape = [2, 4, 4, 5]
        self.out_h = 6
        self.out_w = 7
        self.scale = 0.
        self.data_layout = "NHWC"


class TestBilinearNeighborInterpCase3(TestBilinearInterpMKLDNNOp):

    def init_test_case(self):
        self.input_shape = [1, 1, 32, 64]
        self.out_h = 64
        self.out_w = 128
        self.scale = 0.


class TestBilinearNeighborInterpCase4(TestBilinearInterpMKLDNNOp):

    def init_test_case(self):
        self.input_shape = [4, 1, 7, 8]
        self.out_h = 1
        self.out_w = 1
        self.scale = 0.
        self.out_size = np.array([2, 2]).astype("int32")


class TestBilinearNeighborInterpCase5(TestBilinearInterpMKLDNNOp):

    def init_test_case(self):
        self.input_shape = [1, 1, 9, 6]
        self.out_h = 12
        self.out_w = 12
        self.scale = 0.
        self.out_size = np.array([13, 13]).astype("int32")


class TestBilinearNeighborInterpCase6(TestBilinearInterpMKLDNNOp):

    def init_test_case(self):
        self.input_shape = [1, 1, 32, 64]
        self.out_h = 64
        self.out_w = 32
        self.scale = 0.
        self.out_size = np.array([65, 129]).astype("int32")


class TestBilinearNeighborInterpSame(TestBilinearInterpMKLDNNOp):

    def init_test_case(self):
        self.input_shape = [2, 3, 32, 64]
        self.out_h = 32
        self.out_w = 64
        self.scale = 0.


if __name__ == "__main__":
    from paddle import enable_static
    enable_static()
    unittest.main()
