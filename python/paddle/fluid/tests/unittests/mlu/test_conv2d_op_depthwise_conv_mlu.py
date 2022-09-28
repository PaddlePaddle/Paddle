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
import sys

sys.path.append("..")

import paddle

paddle.enable_static()
import paddle.fluid.core as core
import paddle.fluid as fluid
from op_test import OpTest
from paddle.fluid import Program, program_guard
from test_conv2d_op_mlu import TestConv2DOp, TestConv2DOp_v2, create_test_padding_SAME_class, create_test_padding_VALID_class, create_test_channel_last_class, create_test_fp16_class

#----------------TestDepthwiseConv -----


class TestDepthwiseConv(TestConv2DOp):

    def init_test_case(self):
        self.pad = [1, 1]
        self.stride = [2, 2]
        self.input_size = [2, 3, 5, 5]  # NCHW
        self.groups = 3
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [12, f_c, 3, 3]
        self.op_type = "depthwise_conv2d"


class TestDepthwiseConv2(TestConv2DOp):

    def init_test_case(self):
        self.pad = [1, 1]
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]  # NCHW
        self.groups = 3
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [12, f_c, 3, 3]
        self.op_type = "depthwise_conv2d"


class TestDepthwiseConv3(TestConv2DOp):

    def init_test_case(self):
        self.pad = [1, 1]
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]  # NCHW
        self.groups = 3
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [24, f_c, 3, 3]
        self.op_type = "depthwise_conv2d"


class TestDepthwiseConvandFuse(TestConv2DOp):

    def init_test_case(self):
        self.fuse_relu_before_depthwise_conv = True
        self.pad = [1, 1]
        self.stride = [2, 2]
        self.input_size = [2, 3, 5, 5]  # NCHW
        self.groups = 3
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [12, f_c, 3, 3]
        self.op_type = "depthwise_conv2d"


class TestDepthwiseConv2andFuse(TestConv2DOp):

    def init_test_case(self):
        self.fuse_relu_before_depthwise_conv = True
        self.pad = [1, 1]
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]  # NCHW
        self.groups = 3
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [12, f_c, 3, 3]
        self.op_type = "depthwise_conv2d"


class TestDepthwiseConv3andFuse(TestConv2DOp):

    def init_test_case(self):
        self.fuse_relu_before_depthwise_conv = True
        self.pad = [1, 1]
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]  # NCHW
        self.groups = 3
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [24, f_c, 3, 3]
        self.op_type = "depthwise_conv2d"


class TestDepthwiseConv_AsyPadding(TestConv2DOp_v2):

    def init_test_case(self):
        self.stride = [2, 2]
        self.input_size = [2, 3, 5, 5]  # NCHW
        self.groups = 3
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [12, f_c, 3, 3]
        self.op_type = "depthwise_conv2d"

    def init_paddings(self):
        self.pad = [1, 1, 0, 1]
        self.padding_algorithm = "EXPLICIT"


class TestDepthwiseConv2_AsyPadding(TestConv2DOp_v2):

    def init_test_case(self):
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]  # NCHW
        self.groups = 3
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [12, f_c, 3, 3]
        self.op_type = "depthwise_conv2d"

    def init_paddings(self):
        self.pad = [0, 1, 0, 2]
        self.padding_algorithm = "EXPLICIT"


class TestDepthwiseConv3_AsyPadding(TestConv2DOp_v2):

    def init_test_case(self):
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]  # NCHW
        self.groups = 3
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [24, f_c, 3, 3]
        self.op_type = "depthwise_conv2d"

    def init_paddings(self):
        self.pad = [1, 1, 0, 0]
        self.padding_algorithm = "EXPLICIT"


class TestDepthwiseConvandFuse_AsyPadding(TestConv2DOp_v2):

    def init_test_case(self):
        self.fuse_relu_before_depthwise_conv = True
        self.pad = [1, 1]
        self.stride = [2, 2]
        self.input_size = [2, 3, 5, 5]  # NCHW
        self.groups = 3
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [12, f_c, 3, 3]
        self.op_type = "depthwise_conv2d"

    def init_paddings(self):
        self.pad = [2, 1, 2, 3]
        self.padding_algorithm = "EXPLICIT"


class TestDepthwiseConv2andFuse_AsyPadding(TestConv2DOp_v2):

    def init_test_case(self):
        self.fuse_relu_before_depthwise_conv = True
        self.pad = [1, 1]
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]  # NCHW
        self.groups = 3
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [12, f_c, 3, 3]
        self.op_type = "depthwise_conv2d"

    def init_paddings(self):
        self.pad = [1, 1, 1, 2]
        self.padding_algorithm = "EXPLICIT"


class TestDepthwiseConv3andFuse_AsyPadding(TestConv2DOp_v2):

    def init_test_case(self):
        self.fuse_relu_before_depthwise_conv = True
        self.pad = [1, 1]
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]  # NCHW
        self.groups = 3
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [24, f_c, 3, 3]
        self.op_type = "depthwise_conv2d"

    def init_paddings(self):
        self.pad = [1, 2, 0, 2]
        self.padding_algorithm = "EXPLICIT"


# depthwise conv2d

create_test_padding_SAME_class(TestDepthwiseConv_AsyPadding)
create_test_padding_SAME_class(TestDepthwiseConvandFuse_AsyPadding)

create_test_padding_VALID_class(TestDepthwiseConv_AsyPadding)
create_test_padding_VALID_class(TestDepthwiseConvandFuse_AsyPadding)

# channel last

create_test_channel_last_class(TestDepthwiseConv_AsyPadding)
create_test_channel_last_class(TestDepthwiseConvandFuse_AsyPadding)

create_test_fp16_class(TestDepthwiseConv_AsyPadding)
create_test_fp16_class(TestDepthwiseConvandFuse_AsyPadding)

# TODO(MLU): Depthwise opration does not support dilation yet
# it will throw an error of CNNL_STATUS_NOT_SUPPORTED.

if __name__ == '__main__':
    unittest.main()
