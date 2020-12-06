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
import math
import numpy as np
import paddle.fluid.core as core
import paddle.fluid as fluid
from paddle.fluid.tests.unittests.op_test import OpTest
from paddle.fluid.tests.unittests.op_test import skip_check_grad_ci


def linear_interp_mkldnn_np(input,
                            out_w,
                            out_size=None,
                            actual_shape=None,
                            data_layout='NCHW'):
    if data_layout == "NHWC":
        input = np.transpose(input, (0, 2, 1))  # NHWC => NCHW
    if out_size is not None:
        out_w = out_size[0]
    if actual_shape is not None:
        out_w = actual_shape[0]
    batch_size, channel, in_w = input.shape

    fw = 0.0
    fw = out_w * 1.0 / in_w

    out = np.zeros((batch_size, channel, out_w))

    for ow in range(out_w):
        w0 = int(math.floor((ow + 0.5) / fw - 0.5))
        w0 = min(max(0, w0), in_w - 1)
        w1 = int(math.ceil((ow + 0.5) / fw - 0.5))
        w1 = min(max(0, w1), in_w - 1)
        Ww = (ow + 0.5) / fw - 0.5 - w0

        out[:, :, ow] = input[:, :, w0] * Ww + input[:, :, w1] * (1 - Ww)

    if data_layout == "NHWC":
        out = np.transpose(out, (0, 2, 1))  # NCHW => NHWC

    return out.astype(input.dtype)


@skip_check_grad_ci(reason="Haven not implement interpolate grad kernel.")
class TestLinearInterpMKLDNNOp(OpTest):
    def setUp(self):
        self.out_size = None
        self.actual_shape = None
        self.data_layout = 'NCHW'
        self.init_test_case()
        self.op_type = "linear_interp"
        input_np = np.random.random(self.input_shape).astype("float32")

        if self.data_layout == "NCHW":
            in_w = self.input_shape[2]
        else:
            in_w = self.input_shape[1]

        if self.scale > 0:
            out_w = int(in_w * self.scale)
        else:
            out_w = self.out_w

        output_np = linear_interp_mkldnn_np(input_np, out_w, self.out_size,
                                            self.actual_shape, self.data_layout)
        self.inputs = {'X': input_np}
        if self.out_size is not None:
            self.inputs['OutSize'] = self.out_size
        if self.actual_shape is not None:
            self.inputs['OutSize'] = self.actual_shape

        self.attrs = {
            'out_w': self.out_w,
            'scale': self.scale,
            'interp_method': self.interp_method,
            'data_layout': self.data_layout,
            'use_mkldnn': self.use_mkldnn
        }
        self.outputs = {'Out': output_np}

    def test_check_output(self):
        self.check_output(check_dygraph=False)

    def init_test_case(self):
        self.interp_method = 'linear'
        self.input_shape = [1, 3, 100]
        self.out_w = 50
        self.scale = 0.
        self.out_size = np.array([50, ]).astype("int32")
        self.use_mkldnn = True


# class TestLinearInterpMKLDNNOpDataLayout(TestLinearInterpMKLDNNOp):
#     def init_test_case(self):
#         self.interp_method = 'linear'
#         self.input_shape = [1, 3, 100]
#         self.out_w = 50
#         self.scale = 0.
#         self.out_size = np.array([50, ]).astype("int32")
#         self.data_layout = 'NHWC'
#         self.use_mkldnn = True


class TestLinearInterpMKLDNNOpScale(TestLinearInterpMKLDNNOp):
    def init_test_case(self):
        self.interp_method = 'linear'
        self.input_shape = [1, 3, 100]
        self.out_w = 50
        self.scale = 0.5
        self.out_size = np.array([50, ]).astype("int32")
        self.use_mkldnn = True


# class TestLinearInterpMKLDNNOpSizeTensor(TestLinearInterpMKLDNNOp):
#     def setUp(self):
#         self.out_size = None
#         self.actual_shape = None
#         self.data_layout = 'NCHW'
#         self.init_test_case()
#         self.op_type = "linear_interp"
#         input_np = np.random.random(self.input_shape).astype("float64")
#         self.shape_by_1Dtensor = False
#         self.scale_by_1Dtensor = False

#         if self.data_layout == "NCHW":
#             in_w = self.input_shape[2]
#         else:
#             in_w = self.input_shape[1]

#         if self.scale > 0:
#             out_w = int(in_w * self.scale)
#         else:
#             out_w = self.out_w

#         output_np = linear_interp_mkldnn_np(input_np, out_w, self.out_size,
#                                      self.actual_shape, self.data_layout)

#         self.inputs = {'X': input_np}
#         if self.out_size is not None and self.shape_by_1Dtensor:
#             self.inputs['OutSize'] = self.out_size
#         elif self.actual_shape is not None and self.shape_by_1Dtensor:
#             self.inputs['OutSize'] = self.actual_shape
#         else:
#             size_tensor = []
#             for index, ele in enumerate(self.out_size):
#                 size_tensor.append(("x" + str(index), np.ones(
#                     (1)).astype('int32') * ele))
#             self.inputs['SizeTensor'] = size_tensor

#         self.attrs = {
#             'out_w': self.out_w,
#             'scale': self.scale,
#             'interp_method': self.interp_method,
#             'data_layout': self.data_layout
#         }
#         self.outputs = {'Out': output_np}

if __name__ == "__main__":
    unittest.main()
