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

import paddle

paddle.enable_static()
import sys

sys.path.append("../../legacy_test")
from test_conv2d_transpose_op import TestConv2DTransposeOp


class TestDepthwiseConvTranspose(TestConv2DTransposeOp):
    def init_test_case(self):
        self.pad = [1, 1]
        self.stride = [1, 1]
        self.dilations = [1, 1]
        self.input_size = [1, 8, 4, 4]  # NCHW
        self.groups = 8
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [self.input_size[1], f_c, 4, 4]
        self.op_type = "depthwise_conv2d_transpose"


class TestDepthwiseConvTransposeAsymmetricPad(TestConv2DTransposeOp):
    def init_test_case(self):
        self.pad = [1, 1, 1, 2]
        self.stride = [1, 1]
        self.dilations = [1, 1]
        self.input_size = [1, 8, 4, 4]  # NCHW
        self.groups = 8
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [self.input_size[1], f_c, 3, 3]
        self.op_type = "depthwise_conv2d_transpose"
        self.data_format = 'NCHW'


class TestDepthwiseConvTransposeSAMEPad(TestConv2DTransposeOp):
    def init_test_case(self):
        self.stride = [1, 1]
        self.dilations = [1, 1]
        self.input_size = [1, 8, 4, 4]  # NHWC
        self.groups = 8
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [self.input_size[1], f_c, 3, 3]
        self.op_type = "depthwise_conv2d_transpose"
        self.padding_algorithm = 'SAME'


class TestDepthwiseConvTransposeVALIDPad(TestConv2DTransposeOp):
    def init_test_case(self):
        self.stride = [1, 1]
        self.dilations = [1, 1]
        self.input_size = [1, 8, 4, 4]  # NHWC
        self.groups = 8
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [self.input_size[1], f_c, 3, 3]
        self.op_type = "depthwise_conv2d_transpose"
        self.padding_algorithm = 'VALID'


class TestDepthwiseConvTranspose_NHWC_3x3kernel(TestConv2DTransposeOp):
    def init_test_case(self):
        self.pad = [1, 1]
        self.stride = [1, 1]
        self.dilations = [1, 1]
        self.input_size = [1, 4, 4, 8]  # NHWC
        self.groups = 8
        assert np.mod(self.input_size[3], self.groups) == 0
        f_c = self.input_size[3] // self.groups
        self.filter_size = [self.input_size[3], f_c, 3, 3]
        self.op_type = "depthwise_conv2d_transpose"
        self.data_format = 'NHWC'


if __name__ == '__main__':
    unittest.main()
