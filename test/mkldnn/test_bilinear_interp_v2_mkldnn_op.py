#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import math
import unittest

import numpy as np
from op_test import OpTest, convert_float_to_uint16, skip_check_grad_ci


def bilinear_interp_onednn_np(
    input, out_h, out_w, out_size=None, actual_shape=None, data_layout='NCHW'
):
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
            out[:, :, oh, ow] = (
                input_h0_w0 * (1 - Wh) * (1 - Ww)
                + input_h1_w0 * Wh * (1 - Ww)
                + input_h0_w1 * (1 - Wh) * Ww
                + input_h1_w1 * Wh * Ww
            )

    if data_layout == "NHWC":
        out = np.transpose(out, (0, 2, 3, 1))  # NCHW => NHWC

    return out.astype(input.dtype)


@skip_check_grad_ci(reason="Haven not implement interpolate grad kernel.")
class TestBilinearInterpOneDNNOp(OpTest):
    def init_test_case(self):
        pass

    def init_data_type(self):
        pass

    def setUp(self):
        self.op_type = "bilinear_interp_v2"
        self.interp_method = 'bilinear'
        self._cpu_only = True
        self.use_onednn = True
        self.input_shape = [1, 1, 2, 2]
        self.data_layout = 'NCHW'
        self.dtype = np.float32
        # priority: actual_shape > out_size > scale > out_h & out_w
        self.out_h = 1
        self.out_w = 1
        self.scale = 2.0
        self.out_size = None
        self.actual_shape = None

        self.init_test_case()
        self.init_data_type()

        input_np = np.random.random(self.input_shape).astype(self.dtype)
        if self.dtype == np.uint16:
            input_np = convert_float_to_uint16(input_np)

        if self.data_layout == "NCHW":
            in_h = self.input_shape[2]
            in_w = self.input_shape[3]
        else:
            in_h = self.input_shape[1]
            in_w = self.input_shape[2]

        scale_h = 0
        scale_w = 0

        if self.scale:
            if isinstance(self.scale, (float, int)):
                scale_h = float(self.scale)
                scale_w = float(self.scale)
            if isinstance(self.scale, list) and len(self.scale) == 1:
                scale_w = self.scale[0]
                scale_h = self.scale[0]
            elif isinstance(self.scale, list) and len(self.scale) > 1:
                scale_w = self.scale[1]
                scale_h = self.scale[0]

        if scale_h > 0 and scale_w > 0:
            out_h = int(in_h * scale_h)
            out_w = int(in_w * scale_w)
        else:
            out_h = self.out_h
            out_w = self.out_w

        output_np = bilinear_interp_onednn_np(
            input_np,
            out_h,
            out_w,
            self.out_size,
            self.actual_shape,
            self.data_layout,
        )

        if isinstance(self.scale, float):
            self.scale = [self.scale, self.scale]

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
            'use_mkldnn': self.use_onednn,
        }
        self.outputs = {'Out': output_np}

    def test_check_output(self):
        self.check_output(check_dygraph=False, check_pir_onednn=True)


class TestBilinearInterpOpOneDNNNHWC(TestBilinearInterpOneDNNOp):
    def init_test_case(self):
        self.input_shape = [3, 2, 32, 16]
        self.out_h = 27
        self.out_w = 49
        self.scale = [2.0, 3.0]
        self.data_layout = 'NHWC'


class TestBilinearNeighborInterpOneDNNCase2(TestBilinearInterpOneDNNOp):
    def init_test_case(self):
        self.input_shape = [3, 3, 9, 6]
        self.out_h = 12
        self.out_w = 12


class TestBilinearNeighborInterpOneDNNCase3(TestBilinearInterpOneDNNOp):
    def init_test_case(self):
        self.input_shape = [1, 1, 32, 64]
        self.out_h = 64
        self.out_w = 128
        self.scale = [0.1, 0.05]


class TestBilinearNeighborInterpOneDNNCase4(TestBilinearInterpOneDNNOp):
    def init_test_case(self):
        self.input_shape = [1, 1, 32, 64]
        self.out_h = 64
        self.out_w = 32
        self.scale = [13.0, 15.0]
        self.out_size = np.array([65, 129]).astype("int32")


class TestBilinearNeighborInterpOneDNNCase5(TestBilinearInterpOneDNNOp):
    def init_test_case(self):
        self.input_shape = [1, 1, 9, 6]
        self.out_h = 12
        self.out_w = 12
        self.out_size = np.array([13, 13]).astype("int32")


class TestBilinearNeighborInterpOneDNNCase6(TestBilinearInterpOneDNNOp):
    def init_test_case(self):
        self.input_shape = [1, 1, 32, 64]
        self.out_h = 64
        self.out_w = 32
        self.scale = 1.0
        self.out_size = np.array([65, 129]).astype("int32")


class TestBilinearNeighborInterpOneDNNSame(TestBilinearInterpOneDNNOp):
    def init_test_case(self):
        self.input_shape = [2, 3, 32, 64]
        self.out_h = 32
        self.out_w = 64
        self.scale = 2.0
        self.out_size = np.array([65, 129]).astype("int32")


def create_test_class(parent):
    class TestBf16Case(parent):
        def init_data_type(self):
            self.dtype = np.uint16

    TestBf16Case.__name__ = "{}_{}".format(parent.__name__, "BF16")
    globals()[TestBf16Case.__name__] = TestBf16Case


create_test_class(TestBilinearInterpOneDNNOp)
create_test_class(TestBilinearInterpOpOneDNNNHWC)
create_test_class(TestBilinearNeighborInterpOneDNNCase2)
create_test_class(TestBilinearNeighborInterpOneDNNCase3)
create_test_class(TestBilinearNeighborInterpOneDNNCase4)
create_test_class(TestBilinearNeighborInterpOneDNNCase5)
create_test_class(TestBilinearNeighborInterpOneDNNCase6)
create_test_class(TestBilinearNeighborInterpOneDNNSame)

if __name__ == "__main__":
    from paddle import enable_static

    enable_static()
    unittest.main()
