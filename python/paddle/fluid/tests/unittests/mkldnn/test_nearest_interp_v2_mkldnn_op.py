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

import unittest
import numpy as np
from paddle.fluid.tests.unittests.op_test import OpTest, OpTestTool, convert_float_to_uint16
from paddle.fluid.tests.unittests.op_test import skip_check_grad_ci


def nearest_neighbor_interp_mkldnn_np(X,
                                      out_h,
                                      out_w,
                                      out_size=None,
                                      actual_shape=None,
                                      data_layout='NCHW'):
    """nearest neighbor interpolation implement in shape [N, C, H, W]"""
    if data_layout == "NHWC":
        X = np.transpose(X, (0, 3, 1, 2))  # NHWC => NCHW
    if out_size is not None:
        out_h = out_size[0]
        out_w = out_size[1]
    if actual_shape is not None:
        out_h = actual_shape[0]
        out_w = actual_shape[1]

    n, c, in_h, in_w = X.shape

    fh = fw = 0.0
    if (out_h > 1):
        fh = out_h * 1.0 / in_h
    if (out_w > 1):
        fw = out_w * 1.0 / in_w

    out = np.zeros((n, c, out_h, out_w))

    for oh in range(out_h):
        ih = int(round((oh + 0.5) / fh - 0.5))
        for ow in range(out_w):
            iw = int(round((ow + 0.5) / fw - 0.5))
            out[:, :, oh, ow] = X[:, :, ih, iw]

    if data_layout == "NHWC":
        out = np.transpose(out, (0, 2, 3, 1))  # NCHW => NHWC

    return out.astype(X.dtype)


@skip_check_grad_ci(reason="Haven not implement interpolate grad kernel.")
@OpTestTool.skip_if_not_cpu_bf16()
class TestNearestInterpV2MKLDNNOp(OpTest):

    def init_test_case(self):
        pass

    def init_data_type(self):
        pass

    def setUp(self):
        self.op_type = "nearest_interp_v2"
        self.interp_method = 'nearest'
        self._cpu_only = True
        self.use_mkldnn = True
        self.input_shape = [1, 1, 2, 2]
        self.data_layout = 'NCHW'
        self.dtype = np.float32
        # priority: actual_shape > out_size > scale > out_h & out_w
        self.out_h = 1
        self.out_w = 1
        self.scale = [2.0, 3.0]
        self.out_size = None
        self.actual_shape = None

        self.init_test_case()
        self.init_data_type()

        if self.dtype == np.float32 or self.dtype == np.uint16:
            input_np = np.random.random(self.input_shape).astype(self.dtype)
        else:
            init_low, init_high = (-5, 5) if self.dtype == np.int8 else (0, 10)
            input_np = np.random.randint(init_low, init_high,
                                         self.input_shape).astype(self.dtype)

        if self.data_layout == "NCHW":
            in_h = self.input_shape[2]
            in_w = self.input_shape[3]
        else:
            in_h = self.input_shape[1]
            in_w = self.input_shape[2]

        scale_h = 0
        scale_w = 0

        if self.scale:
            if isinstance(self.scale, float) or isinstance(self.scale, int):
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

        output_np = nearest_neighbor_interp_mkldnn_np(input_np, out_h, out_w,
                                                      self.out_size,
                                                      self.actual_shape,
                                                      self.data_layout)

        if isinstance(self.scale, float):
            self.scale = [self.scale]

        if self.dtype == np.uint16:
            input_np = convert_float_to_uint16(input_np)

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


class TestNearestInterpOpV2MKLDNNNHWC(TestNearestInterpV2MKLDNNOp):

    def init_test_case(self):
        self.input_shape = [3, 2, 32, 16]
        self.out_h = 27
        self.out_w = 49
        self.scale = [2.0, 3.0]
        self.data_layout = 'NHWC'


class TestNearestNeighborInterpV2MKLDNNCase2(TestNearestInterpV2MKLDNNOp):

    def init_test_case(self):
        self.input_shape = [3, 3, 9, 6]
        self.out_h = 12
        self.out_w = 12


class TestNearestNeighborInterpV2MKLDNNCase3(TestNearestInterpV2MKLDNNOp):

    def init_test_case(self):
        self.input_shape = [1, 1, 32, 64]
        self.out_h = 64
        self.out_w = 128
        self.scale = [0.1, 0.05]


class TestNearestNeighborInterpV2MKLDNNCase4(TestNearestInterpV2MKLDNNOp):

    def init_test_case(self):
        self.input_shape = [1, 1, 32, 64]
        self.out_h = 64
        self.out_w = 32
        self.scale = [13.0, 15.0]
        self.out_size = np.array([65, 129]).astype("int32")


class TestNearestNeighborInterpV2MKLDNNSame(TestNearestInterpV2MKLDNNOp):

    def init_test_case(self):
        self.input_shape = [2, 3, 32, 64]
        self.out_h = 32
        self.out_w = 64
        self.out_size = np.array([65, 129]).astype("int32")


def create_test_class(parent):

    class TestFp32Case(parent):

        def init_data_type(self):
            self.dtype = np.float32

    class TestBf16Case(parent):

        def init_data_type(self):
            self.dtype = np.uint16

    class TestInt8Case(parent):

        def init_data_type(self):
            self.dtype = np.int8

    class TestUint8Case(parent):

        def init_data_type(self):
            self.dtype = np.uint8

    TestFp32Case.__name__ = "{0}_{1}".format(parent.__name__, "FP32")
    TestBf16Case.__name__ = "{0}_{1}".format(parent.__name__, "BF16")
    TestInt8Case.__name__ = "{0}_{1}".format(parent.__name__, "INT8")
    TestUint8Case.__name__ = "{0}_{1}".format(parent.__name__, "UINT8")
    globals()[TestFp32Case.__name__] = TestFp32Case
    globals()[TestBf16Case.__name__] = TestBf16Case
    globals()[TestInt8Case.__name__] = TestInt8Case
    globals()[TestUint8Case.__name__] = TestUint8Case


create_test_class(TestNearestInterpV2MKLDNNOp)
create_test_class(TestNearestInterpOpV2MKLDNNNHWC)
create_test_class(TestNearestNeighborInterpV2MKLDNNCase2)
create_test_class(TestNearestNeighborInterpV2MKLDNNCase3)
create_test_class(TestNearestNeighborInterpV2MKLDNNCase4)
create_test_class(TestNearestNeighborInterpV2MKLDNNSame)

if __name__ == "__main__":
    from paddle import enable_static
    enable_static()
    unittest.main()
