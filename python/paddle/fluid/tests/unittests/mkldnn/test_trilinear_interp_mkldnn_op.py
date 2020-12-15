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
from paddle.fluid.tests.unittests.op_test import OpTest
import math
import paddle.fluid.core as core
import paddle.fluid as fluid
from paddle.nn.functional import interpolate
from paddle.fluid.tests.unittests.op_test import skip_check_grad_ci


def trilinear_interp_mkldnn_np(input,
                               out_d,
                               out_h,
                               out_w,
                               out_size=None,
                               actual_shape=None,
                               data_layout='NCDHW'):
    """trilinear interpolation implement in shape [N, C, D, H, W]"""
    if data_layout == "NDHWC":
        input = np.transpose(input, (0, 4, 1, 2, 3))  # NDHWC => NCDHW
    if out_size is not None:
        out_d = out_size[0]
        out_h = out_size[1]
        out_w = out_size[2]
    if actual_shape is not None:
        out_d = actual_shape[0]
        out_h = actual_shape[1]
        out_w = actual_shape[2]
    batch_size, channel, in_d, in_h, in_w = input.shape

    out = np.zeros((batch_size, channel, out_d, out_h, out_w))

    for od in range(out_d):
        d0 = int(math.floor((od + 0.5) * in_d / out_d - 0.5))
        d0 = max(0, d0)
        d1 = int(math.ceil((od + 0.5) * in_d / out_d - 0.5))
        d1 = min(d1, in_d - 1)
        d0lambda = (od + 0.5) * in_d / out_d - 0.5 - d0
        d1lambda = 1 - d0lambda
        for oh in range(out_h):
            h0 = int(math.floor((oh + 0.5) * in_h / out_h - 0.5))
            h1 = int(math.ceil((oh + 0.5) * in_h / out_h - 0.5))
            h0 = max(h0, 0)
            h1 = min(h1, in_h - 1)
            h0lambda = (oh + 0.5) * in_h / out_h - 0.5 - h0
            h1lambda = 1 - h0lambda
            for ow in range(out_w):
                w0 = int(math.floor((ow + 0.5) * in_w / out_w - 0.5))
                w1 = int(math.ceil((ow + 0.5) * in_w / out_w - 0.5))
                w0 = max(w0, 0)
                w1 = min(w1, in_w - 1)
                w0lambda = (ow + 0.5) * in_w / out_w - 0.5 - w0
                w1lambda = 1 - w0lambda


                out[:, :, od, oh, ow] = \
                    (d1lambda) * \
                    (h1lambda * (w1lambda * input[:, :, d0, h0, w0] + \
                            w0lambda * input[:, :, d0, h0, w1]) + \
                    h0lambda * (w1lambda * input[:, :, d0, h1, w0] + \
                            w0lambda * input[:, :, d0, h1, w1])) + \
                    d0lambda * \
                    (h1lambda * (w1lambda * input[:, :, d1, h0, w0] + \
                            w0lambda * input[:, :, d1, h0, w1]) + \
                    h0lambda * (w1lambda * input[:, :, d1, h1, w0] + \
                            w0lambda * input[:, :, d1, h1, w1]))

    if data_layout == "NDHWC":
        out = np.transpose(out, (0, 2, 3, 4, 1))  # NCDHW => NDHWC

    return out.astype(input.dtype)


@skip_check_grad_ci(reason="Haven not implement interpolate grad kernel.")
class TestTrilinearInterpMKLDNNOp(OpTest):
    def setUp(self):
        self.use_mkldnn = True
        self.out_size = None
        self.actual_shape = None
        self.data_layout = 'NCDHW'
        self.init_test_case()
        self.op_type = "trilinear_interp"
        input_np = np.random.random(self.input_shape).astype("float32")

        if self.data_layout == "NCDHW":
            in_d = self.input_shape[2]
            in_h = self.input_shape[3]
            in_w = self.input_shape[4]
        else:
            in_d = self.input_shape[1]
            in_h = self.input_shape[2]
            in_w = self.input_shape[3]

        if self.scale > 0:
            out_d = int(in_d * self.scale)
            out_h = int(in_h * self.scale)
            out_w = int(in_w * self.scale)
        else:
            out_d = self.out_d
            out_h = self.out_h
            out_w = self.out_w

        output_np = trilinear_interp_mkldnn_np(input_np, out_d, out_h, out_w,
                                               self.out_size, self.actual_shape,
                                               self.data_layout)
        self.inputs = {'X': input_np}
        if self.out_size is not None:
            self.inputs['OutSize'] = self.out_size
        if self.actual_shape is not None:
            self.inputs['OutSize'] = self.actual_shape
        # c++ end treat NCDHW the same way as NCHW
        if self.data_layout == 'NCDHW':
            data_layout = 'NCHW'
        else:
            data_layout = 'NHWC'
        self.attrs = {
            'out_d': self.out_d,
            'out_h': self.out_h,
            'out_w': self.out_w,
            'scale': self.scale,
            'interp_method': self.interp_method,
            'data_layout': data_layout,
            'use_mkldnn': self.use_mkldnn
        }
        self.outputs = {'Out': output_np}

    def test_check_output(self):
        self.check_output(check_dygraph=False)

    def init_test_case(self):
        self.interp_method = 'trilinear'
        self.input_shape = [1, 1, 3, 4, 5]
        self.out_d = 2
        self.out_h = 2
        self.out_w = 2
        self.scale = 0.
        self.use_mkldnn = True
        self.out_size = np.array([3, 3, 3]).astype("int32")


class TestTrilinearInterpCase1(TestTrilinearInterpMKLDNNOp):
    def init_test_case(self):
        self.interp_method = 'trilinear'
        self.input_shape = [2, 1, 7, 8, 9]
        self.out_d = 1
        self.out_h = 1
        self.out_w = 1
        self.scale = 0.
        self.use_mkldnn = True


class TestTrilinearInterpCase2(TestTrilinearInterpMKLDNNOp):
    def init_test_case(self):
        self.interp_method = 'trilinear'
        self.input_shape = [2, 3, 9, 6, 8]
        self.out_d = 12
        self.out_h = 12
        self.out_w = 12
        self.scale = 0.
        self.use_mkldnn = True


class TestTrilinearInterpCase3(TestTrilinearInterpMKLDNNOp):
    def init_test_case(self):
        self.interp_method = 'trilinear'
        self.input_shape = [3, 2, 16, 8, 4]
        self.out_d = 32
        self.out_h = 16
        self.out_w = 8
        self.scale = 0.
        self.use_mkldnn = True


class TestTrilinearInterpCase4(TestTrilinearInterpMKLDNNOp):
    def init_test_case(self):
        self.interp_method = 'trilinear'
        self.input_shape = [4, 1, 7, 8, 9]
        self.out_d = 1
        self.out_h = 1
        self.out_w = 1
        self.scale = 0.
        self.out_size = np.array([2, 2, 2]).astype("int32")
        self.use_mkldnn = True


class TestTrilinearInterpCase5(TestTrilinearInterpMKLDNNOp):
    def init_test_case(self):
        self.interp_method = 'trilinear'
        self.input_shape = [3, 3, 9, 6, 8]
        self.out_d = 12
        self.out_h = 12
        self.out_w = 12
        self.scale = 0.
        self.out_size = np.array([11, 11, 11]).astype("int32")
        self.use_mkldnn = True


class TestTrilinearInterpCase6(TestTrilinearInterpMKLDNNOp):
    def init_test_case(self):
        self.interp_method = 'trilinear'
        self.input_shape = [1, 1, 16, 8, 4]
        self.out_d = 8
        self.out_h = 32
        self.out_w = 16
        self.scale = 0.
        self.out_size = np.array([17, 9, 5]).astype("int32")
        self.use_mkldnn = True


class TestTrilinearInterpSame(TestTrilinearInterpMKLDNNOp):
    def init_test_case(self):
        self.interp_method = 'trilinear'
        self.input_shape = [1, 1, 16, 8, 4]
        self.out_d = 16
        self.out_h = 8
        self.out_w = 4
        self.scale = 0.
        self.use_mkldnn = True


class TestTrilinearInterpSameHW(TestTrilinearInterpMKLDNNOp):
    def init_test_case(self):
        self.interp_method = 'trilinear'
        self.input_shape = [1, 1, 16, 8, 4]
        self.out_d = 8
        self.out_h = 8
        self.out_w = 4
        self.scale = 0.
        self.use_mkldnn = True


class TestTrilinearInterpActualShape(TestTrilinearInterpMKLDNNOp):
    def init_test_case(self):
        self.interp_method = 'trilinear'
        self.input_shape = [3, 2, 16, 8, 4]
        self.out_d = 64
        self.out_h = 32
        self.out_w = 16
        self.scale = 0.
        self.out_size = np.array([33, 19, 7]).astype("int32")
        self.use_mkldnn = True


@skip_check_grad_ci(reason="Haven not implement interpolate grad kernel.")
class TestTrilinearInterpDatalayout(TestTrilinearInterpMKLDNNOp):
    def init_test_case(self):
        self.interp_method = 'trilinear'
        self.input_shape = [2, 4, 4, 4, 3]
        self.out_d = 2
        self.out_h = 2
        self.out_w = 2
        self.scale = 0.
        self.out_size = np.array([3, 3, 3]).astype("int32")
        self.data_layout = "NDHWC"


class TestTrilinearInterpScale1(TestTrilinearInterpMKLDNNOp):
    def init_test_case(self):
        self.interp_method = 'trilinear'
        self.input_shape = [2, 3, 5, 7, 9]
        self.out_d = 82
        self.out_h = 60
        self.out_w = 25
        self.scale = 2.
        self.use_mkldnn = True


class TestTrilinearInterpScale2(TestTrilinearInterpMKLDNNOp):
    def init_test_case(self):
        self.interp_method = 'trilinear'
        self.input_shape = [2, 3, 5, 7, 9]
        self.out_d = 60
        self.out_h = 40
        self.out_w = 25
        self.scale = 1.
        self.use_mkldnn = True


class TestTrilinearInterpScale3(TestTrilinearInterpMKLDNNOp):
    def init_test_case(self):
        self.interp_method = 'trilinear'
        self.input_shape = [2, 3, 5, 7, 9]
        self.out_d = 60
        self.out_h = 40
        self.out_w = 25
        self.scale = 1.5
        self.use_mkldnn = True


class TestTrilinearInterpZero(TestTrilinearInterpMKLDNNOp):
    def init_test_case(self):
        self.interp_method = 'trilinear'
        self.input_shape = [2, 3, 5, 7, 11]
        self.out_d = 60
        self.out_h = 40
        self.out_w = 25
        self.scale = 0.2
        self.use_mkldnn = True


@skip_check_grad_ci(reason="Haven not implement interpolate grad kernel.")
class TestTrilinearInterpOp_attr_tensor(OpTest):
    def setUp(self):
        self.use_mkldnn = True
        self.out_size = None
        self.actual_shape = None
        self.init_test_case()
        self.op_type = "trilinear_interp"
        self.shape_by_1Dtensor = False
        self.scale_by_1Dtensor = False
        self.attrs = {'interp_method': self.interp_method, }

        input_np = np.random.random(self.input_shape).astype("float32")
        self.inputs = {'X': input_np}

        if self.scale_by_1Dtensor:
            self.inputs['Scale'] = np.array([self.scale]).astype("float32")
        elif self.scale > 0:
            out_d = int(self.input_shape[2] * self.scale)
            out_h = int(self.input_shape[3] * self.scale)
            out_w = int(self.input_shape[4] * self.scale)
            self.attrs['scale'] = self.scale
        else:
            out_d = self.out_d
            out_h = self.out_h
            out_w = self.out_w

        if self.shape_by_1Dtensor:
            self.inputs['OutSize'] = self.out_size
        elif self.out_size is not None:
            size_tensor = []
            for index, ele in enumerate(self.out_size):
                size_tensor.append(("x" + str(index), np.ones(
                    (1)).astype('int32') * ele))
            self.inputs['SizeTensor'] = size_tensor

        self.attrs['out_d'] = self.out_d
        self.attrs['out_h'] = self.out_h
        self.attrs['out_w'] = self.out_w
        self.attrs['use_mkldnn'] = self.use_mkldnn
        output_np = trilinear_interp_mkldnn_np(input_np, out_d, out_h, out_w,
                                               self.out_size, self.actual_shape)
        self.outputs = {'Out': output_np}

    def test_check_output(self):
        self.check_output(check_dygraph=False)

    def init_test_case(self):
        self.interp_method = 'trilinear'
        self.input_shape = [2, 3, 4, 4, 4]
        self.out_d = 2
        self.out_h = 3
        self.out_w = 3
        self.scale = 0.
        self.out_size = [2, 3, 3]
        self.use_mkldnn = True


# # out_size is a 1-D tensor
class TestTrilinearInterp_attr_tensor_Case1(TestTrilinearInterpOp_attr_tensor):
    def init_test_case(self):
        self.interp_method = 'trilinear'
        self.input_shape = [3, 2, 9, 6, 8]
        self.out_d = 32
        self.out_h = 16
        self.out_w = 8
        self.scale = 0.3
        self.out_size = [12, 4, 4]
        self.use_mkldnn = True


# # scale is a 1-D tensor
class TestTrilinearInterp_attr_tensor_Case2(TestTrilinearInterpOp_attr_tensor):
    def init_test_case(self):
        self.interp_method = 'trilinear'
        self.input_shape = [2, 3, 8, 8, 4]
        self.out_d = 16
        self.out_h = 12
        self.out_w = 4
        self.scale = 0.
        self.out_size = [16, 4, 10]
        self.shape_by_1Dtensor = True
        self.use_mkldnn = True


# # scale is a 1-D tensor
class TestTrilinearInterp_attr_tensor_Case3(TestTrilinearInterpOp_attr_tensor):
    def init_test_case(self):
        self.interp_method = 'trilinear'
        self.input_shape = [2, 3, 8, 8, 4]
        self.out_d = 16
        self.out_h = 16
        self.out_w = 8
        self.scale = 2.0
        self.out_size = None
        self.scale_by_1Dtensor = True
        self.use_mkldnn = True


if __name__ == "__main__":
    unittest.main()
