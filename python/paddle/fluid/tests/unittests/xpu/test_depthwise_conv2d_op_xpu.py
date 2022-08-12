#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import sys

sys.path.append("..")
import unittest
import numpy as np

import paddle

paddle.enable_static()
import paddle.fluid.core as core
import paddle.fluid as fluid
from op_test_xpu import XPUOpTest
from paddle.fluid import Program, program_guard
from test_conv2d_op_xpu import XPUTestConv2DOp, XPUTestConv2DOp_v2
from xpu.get_test_cover_info import create_test_class, get_xpu_op_support_types, XPUOpTestWrapper


class XPUTestDepthwiseConv2DOp(XPUOpTestWrapper):

    def __init__(self):
        self.op_name = 'depthwise_conv2d'
        self.use_dynamic_create_class = False

    class TestDepthwiseConv(XPUTestConv2DOp.TestConv2DOp):

        def init_test_case(self):
            self.use_cuda = False
            self.pad = [1, 1]
            self.stride = [2, 2]
            self.input_size = [2, 12, 5, 5]  # NCHW
            self.groups = 12
            assert np.mod(self.input_size[1], self.groups) == 0
            f_c = self.input_size[1] // self.groups
            self.filter_size = [12, f_c, 3, 3]
            self.op_type = "depthwise_conv2d"

    class TestDepthwiseConv2(XPUTestConv2DOp.TestConv2DOp):

        def init_test_case(self):
            self.use_cuda = False
            self.pad = [1, 1]
            self.stride = [1, 1]
            self.input_size = [2, 12, 5, 5]  # NCHW
            self.groups = 12
            assert np.mod(self.input_size[1], self.groups) == 0
            f_c = self.input_size[1] // self.groups
            self.filter_size = [12, f_c, 3, 3]
            self.op_type = "depthwise_conv2d"

    class TestDepthwiseConv3(XPUTestConv2DOp.TestConv2DOp):

        def init_test_case(self):
            self.use_cuda = False
            self.pad = [1, 1]
            self.stride = [1, 1]
            self.input_size = [2, 24, 5, 5]  # NCHW
            self.groups = 24
            assert np.mod(self.input_size[1], self.groups) == 0
            f_c = self.input_size[1] // self.groups
            self.filter_size = [24, f_c, 3, 3]
            self.op_type = "depthwise_conv2d"

    class TestDepthwiseConvWithDilation(XPUTestConv2DOp.TestConv2DOp):

        def init_test_case(self):
            self.use_cuda = False
            self.pad = [1, 1]
            self.stride = [2, 2]
            self.input_size = [2, 24, 5, 5]  # NCHW
            self.groups = 24
            self.dilations = [2, 2]
            assert np.mod(self.input_size[1], self.groups) == 0
            f_c = self.input_size[1] // self.groups
            self.filter_size = [24, f_c, 3, 3]
            self.op_type = "depthwise_conv2d"

    class TestDepthwiseConvWithDilation2(XPUTestConv2DOp.TestConv2DOp):

        def init_test_case(self):
            self.use_cuda = False
            self.pad = [1, 1]
            self.stride = [1, 1]
            self.input_size = [2, 24, 5, 5]  # NCHW
            self.groups = 24
            self.dilations = [2, 2]
            assert np.mod(self.input_size[1], self.groups) == 0
            f_c = self.input_size[1] // self.groups
            self.filter_size = [24, f_c, 3, 3]
            self.op_type = "depthwise_conv2d"


class XPUTestDepthwiseConv2DOp_v2(XPUOpTestWrapper):

    def __init__(self):
        self.op_name = 'depthwise_conv2d'
        self.use_dynamic_create_class = False

    class TestDepthwiseConv_AsyPadding(XPUTestConv2DOp_v2.TestConv2DOp_v2):

        def init_test_case(self):
            self.use_cuda = False
            self.stride = [2, 2]
            self.input_size = [2, 12, 5, 5]  # NCHW
            self.groups = 12
            assert np.mod(self.input_size[1], self.groups) == 0
            f_c = self.input_size[1] // self.groups
            self.filter_size = [12, f_c, 3, 3]
            self.op_type = "depthwise_conv2d"

        def init_paddings(self):
            self.pad = [1, 1, 0, 1]
            self.padding_algorithm = "EXPLICIT"

    class TestDepthwiseConv2_AsyPadding(XPUTestConv2DOp_v2.TestConv2DOp_v2):

        def init_test_case(self):
            self.use_cuda = False
            self.stride = [1, 1]
            self.input_size = [2, 12, 5, 5]  # NCHW
            self.groups = 12
            assert np.mod(self.input_size[1], self.groups) == 0
            f_c = self.input_size[1] // self.groups
            self.filter_size = [12, f_c, 3, 3]
            self.op_type = "depthwise_conv2d"

        def init_paddings(self):
            self.pad = [0, 1, 0, 2]
            self.padding_algorithm = "EXPLICIT"

    class TestDepthwiseConv3_AsyPadding(XPUTestConv2DOp_v2.TestConv2DOp_v2):

        def init_test_case(self):
            self.use_cuda = False
            self.stride = [1, 1]
            self.input_size = [2, 24, 5, 5]  # NCHW
            self.groups = 24
            assert np.mod(self.input_size[1], self.groups) == 0
            f_c = self.input_size[1] // self.groups
            self.filter_size = [24, f_c, 3, 3]
            self.op_type = "depthwise_conv2d"

        def init_paddings(self):
            self.pad = [1, 1, 0, 0]
            self.padding_algorithm = "EXPLICIT"

    class TestDepthwiseConvWithDilation_AsyPadding(
            XPUTestConv2DOp_v2.TestConv2DOp_v2):

        def init_test_case(self):
            self.use_cuda = False
            self.pad = [1, 1]
            self.stride = [2, 2]
            self.input_size = [2, 24, 5, 5]  # NCHW
            self.groups = 24
            self.dilations = [2, 2]
            assert np.mod(self.input_size[1], self.groups) == 0
            f_c = self.input_size[1] // self.groups
            self.filter_size = [24, f_c, 3, 3]
            self.op_type = "depthwise_conv2d"

        def init_paddings(self):
            self.pad = [1, 1, 2, 1]
            self.padding_algorithm = "EXPLICIT"

    class TestDepthwiseConvWithDilation2_AsyPadding(
            XPUTestConv2DOp_v2.TestConv2DOp_v2):

        def init_test_case(self):
            self.use_cuda = True
            self.pad = [1, 1]
            self.stride = [1, 1]
            self.input_size = [2, 24, 5, 5]  # NCHW
            self.groups = 24
            self.dilations = [2, 2]
            assert np.mod(self.input_size[1], self.groups) == 0
            f_c = self.input_size[1] // self.groups
            self.filter_size = [24, f_c, 3, 3]
            self.op_type = "depthwise_conv2d"

        def init_paddings(self):
            self.pad = [0, 1, 1, 0]
            self.padding_algorithm = "EXPLICIT"


support_types = get_xpu_op_support_types('depthwise_conv2d')
for stype in support_types:
    create_test_class(globals(), XPUTestDepthwiseConv2DOp, stype)
    create_test_class(globals(), XPUTestDepthwiseConv2DOp_v2, stype)

#### depthwise conv2d

# create_test_padding_SAME_class(TestDepthwiseConv_AsyPadding)
# create_test_padding_SAME_class(TestDepthwiseConvWithDilation_AsyPadding)
# create_test_padding_SAME_class(TestDepthwiseConvandFuse_AsyPadding)
# create_test_padding_SAME_class(TestDepthwiseConvWithDilationandFuse_AsyPadding)

# create_test_padding_VALID_class(TestDepthwiseConv_AsyPadding)
# create_test_padding_VALID_class(TestDepthwiseConvWithDilation_AsyPadding)
# create_test_padding_VALID_class(TestDepthwiseConvandFuse_AsyPadding)
# create_test_padding_VALID_class(TestDepthwiseConvWithDilationandFuse_AsyPadding)

#### channel last

# create_test_channel_last_class(TestDepthwiseConv_AsyPadding)
# create_test_channel_last_class(TestDepthwiseConvWithDilation2_AsyPadding)
# create_test_channel_last_class(TestDepthwiseConvandFuse_AsyPadding)
# create_test_channel_last_class(TestDepthwiseConvWithDilationandFuse_AsyPadding)

if __name__ == '__main__':
    unittest.main()
