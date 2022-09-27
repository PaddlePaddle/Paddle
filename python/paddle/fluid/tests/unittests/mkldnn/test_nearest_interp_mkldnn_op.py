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
from paddle.fluid.tests.unittests.op_test import OpTest
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
class TestNearestInterpMKLDNNOp(OpTest):

    def init_test_case(self):
        pass

    def init_data_type(self):
        pass

    def setUp(self):
        self.op_type = "nearest_interp"
        self.interp_method = 'nearest'
        self._cpu_only = True
        self.use_mkldnn = True
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

        if self.dtype == np.float32:
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

        if self.scale > 0:
            out_h = int(in_h * self.scale)
            out_w = int(in_w * self.scale)
        else:
            out_h = self.out_h
            out_w = self.out_w

        output_np = nearest_neighbor_interp_mkldnn_np(input_np, out_h, out_w,
                                                      self.out_size,
                                                      self.actual_shape,
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


class TestNearestInterpOpMKLDNNNHWC(TestNearestInterpMKLDNNOp):

    def init_test_case(self):
        self.input_shape = [3, 2, 32, 16]
        self.out_h = 27
        self.out_w = 49
        self.scale = 2.0
        self.data_layout = 'NHWC'


class TestNearestNeighborInterpMKLDNNCase2(TestNearestInterpMKLDNNOp):

    def init_test_case(self):
        self.input_shape = [3, 3, 9, 6]
        self.out_h = 12
        self.out_w = 12
        self.scale = 1.


class TestNearestNeighborInterpCase3(TestNearestInterpMKLDNNOp):

    def init_test_case(self):
        self.input_shape = [1, 1, 32, 64]
        self.out_h = 64
        self.out_w = 128
        self.scale = 0.


class TestNearestNeighborInterpCase4(TestNearestInterpMKLDNNOp):

    def init_test_case(self):
        self.input_shape = [1, 1, 32, 64]
        self.out_h = 64
        self.out_w = 32
        self.scale = 0.
        self.out_size = np.array([65, 129]).astype("int32")


class TestNearestNeighborInterpSame(TestNearestInterpMKLDNNOp):

    def init_test_case(self):
        self.input_shape = [2, 3, 32, 64]
        self.out_h = 32
        self.out_w = 64
        self.scale = 0.


def create_test_class(parent):

    class TestFp32Case(parent):

        def init_data_type(self):
            self.dtype = np.float32

    class TestInt8Case(parent):

        def init_data_type(self):
            self.dtype = np.int8

    class TestUint8Case(parent):

        def init_data_type(self):
            self.dtype = np.uint8

    TestFp32Case.__name__ = parent.__name__
    TestInt8Case.__name__ = parent.__name__
    TestUint8Case.__name__ = parent.__name__
    globals()[parent.__name__] = TestFp32Case
    globals()[parent.__name__] = TestInt8Case
    globals()[parent.__name__] = TestUint8Case


create_test_class(TestNearestInterpMKLDNNOp)
create_test_class(TestNearestInterpOpMKLDNNNHWC)
create_test_class(TestNearestNeighborInterpMKLDNNCase2)
create_test_class(TestNearestNeighborInterpCase3)
create_test_class(TestNearestNeighborInterpCase4)
create_test_class(TestNearestInterpOpMKLDNNNHWC)
create_test_class(TestNearestNeighborInterpSame)

if __name__ == "__main__":
    from paddle import enable_static
    enable_static()
    unittest.main()
