# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.fluid as fluid
import sys
sys.path.append("..")
from op_test import OpTest, skip_check_grad_ci
from test_conv2d_op import conv2d_forward_naive

paddle.enable_static()


def create_test_channel_last_class(parent):
    class TestChannelLastCase(parent):
        def init_data_format(self):
            self.data_format = "NHWC"

        def init_test_case_2(self):
            N, C, H, W = self.input_size
            self.input_size = [N, H, W, C]

    cls_name = "{0}_{1}".format(parent.__name__, "ChannelLast")
    TestChannelLastCase.__name__ = cls_name
    globals()[cls_name] = TestChannelLastCase


def create_test_padding_SAME_class(parent):
    class TestPaddingSMAECase(parent):
        def init_paddings(self):
            self.pad = [0, 0]
            self.padding_algorithm = "SAME"

    cls_name = "{0}_{1}".format(parent.__name__, "PaddingSAMEOp")
    TestPaddingSMAECase.__name__ = cls_name
    globals()[cls_name] = TestPaddingSMAECase


def create_test_padding_VALID_class(parent):
    class TestPaddingVALIDCase(parent):
        def init_paddings(self):
            self.pad = [1, 1]
            self.padding_algorithm = "VALID"

    cls_name = "{0}_{1}".format(parent.__name__, "PaddingVALIDOp")
    TestPaddingVALIDCase.__name__ = cls_name
    globals()[cls_name] = TestPaddingVALIDCase


@skip_check_grad_ci(
    reason='''Inference only, it doesn't need to call check_grad.''')
class TestDepthwiseConvNPU(OpTest):
    def setUp(self):
        self.op_type = "depthwise_conv2d"
        self.dtype = np.float16
        self.set_npu()
        self.init_data_format()
        self.init_test_case()
        self.init_test_case_2()

        conv2d_param = {
            'stride': self.stride,
            'pad': self.pad,
            'dilation': self.dilations
        }

        input = np.random.random(self.input_size).astype(self.dtype)
        filter = np.random.uniform(-1, 1, self.filter_size).astype(self.dtype)

        output, _, _, _, _ = conv2d_forward_naive(input, filter, self.groups,
                                                  conv2d_param, "EXPLICIT",
                                                  self.data_format)

        output = output.astype(self.dtype)

        self.inputs = {
            'Input': OpTest.np_dtype_to_fluid_dtype(input),
            'Filter': OpTest.np_dtype_to_fluid_dtype(filter)
        }

        self.attrs = {
            'strides': self.stride,
            'paddings': self.pad,
            'groups': self.groups,
            'dilations': self.dilations,
            'data_format': self.data_format,
        }
        self.outputs = {'Output': output}

    def set_npu(self):
        self.__class__.use_npu = True
        self.place = paddle.NPUPlace(0)

    def init_test_case(self):
        self.pad = [1, 1]
        self.dilations = [1, 1]
        self.stride = [2, 2]
        self.input_size = [2, 3, 5, 5]  # NCHW
        self.groups = 3
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [3, f_c, 3, 3]

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def init_data_format(self):
        self.data_format = "NCHW"

    def init_test_case_2(self):
        pass


class TestDepthwiseConvNPU2(TestDepthwiseConvNPU):
    def init_test_case(self):
        self.pad = [1, 1]
        self.dilations = [1, 1]
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]  # NCHW
        self.groups = 3
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [3, f_c, 3, 3]


class TestDepthwiseConvNPU3(TestDepthwiseConvNPU):
    def init_test_case(self):
        self.pad = [1, 1]
        self.dilations = [2, 2]
        self.stride = [2, 2]
        self.input_size = [2, 3, 5, 5]  # NCHW
        self.groups = 3
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [3, f_c, 3, 3]


class TestDepthwiseConvNPU4(TestDepthwiseConvNPU):
    def init_test_case(self):
        self.pad = [1, 1]
        self.dilations = [2, 2]
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]  # NCHW
        self.groups = 3
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [3, f_c, 3, 3]


@skip_check_grad_ci(
    reason='''Inference only, it doesn't need to call check_grad.''')
class TestDepthwiseConvNPU_Padding(OpTest):
    def setUp(self):
        self.op_type = "depthwise_conv2d"
        self.dtype = np.float16
        self.set_npu()
        self.init_data_format()
        self.init_paddings()
        self.init_test_case()
        self.init_test_case_2()

        conv2d_param = {
            'stride': self.stride,
            'pad': self.pad,
            'dilation': self.dilations
        }

        input = np.random.random(self.input_size).astype(self.dtype)
        filter = np.random.uniform(-1, 1, self.filter_size).astype(self.dtype)

        output, _, _, _, _ = conv2d_forward_naive(
            input, filter, self.groups, conv2d_param, self.padding_algorithm,
            self.data_format)
        output = output.astype(self.dtype)

        self.inputs = {
            'Input': OpTest.np_dtype_to_fluid_dtype(input),
            'Filter': OpTest.np_dtype_to_fluid_dtype(filter)
        }

        self.attrs = {
            'strides': self.stride,
            'paddings': self.pad,
            'padding_algorithm': self.padding_algorithm,
            'groups': self.groups,
            'dilations': self.dilations,
            'data_format': self.data_format
        }
        self.outputs = {'Output': output}

    def set_npu(self):
        self.__class__.use_npu = True
        self.place = paddle.NPUPlace(0)

    def init_test_case(self):
        self.pad = [1, 1, 0, 1]
        self.dilations = [1, 1]
        self.stride = [2, 2]
        self.input_size = [2, 3, 5, 5]  # NCHW
        self.groups = 3
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [3, f_c, 3, 3]

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def init_data_format(self):
        self.data_format = "NCHW"

    def init_paddings(self):
        self.pad = [1, 1, 0, 1]
        self.padding_algorithm = "EXPLICIT"

    def init_test_case_2(self):
        pass


class TestDepthwiseConvNPU2_Padding(TestDepthwiseConvNPU_Padding):
    def init_test_case(self):
        self.pad = [1, 1, 0, 1]
        self.dilations = [1, 1]
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]  # NCHW
        self.groups = 3
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [3, f_c, 3, 3]

    def init_paddings(self):
        self.pad = [0, 1, 0, 2]
        self.padding_algorithm = "EXPLICIT"


class TestDepthwiseConvNPU3_Padding(TestDepthwiseConvNPU_Padding):
    def init_test_case(self):
        self.pad = [1, 1, 0, 1]
        self.dilations = [1, 1]
        self.stride = [2, 2]
        self.input_size = [2, 3, 5, 5]  # NCHW
        self.groups = 3
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [3, f_c, 3, 3]

    def init_paddings(self):
        self.pad = [2, 1, 2, 3]
        self.padding_algorithm = "EXPLICIT"


# test channel last
create_test_channel_last_class(TestDepthwiseConvNPU)
create_test_channel_last_class(TestDepthwiseConvNPU2)
create_test_channel_last_class(TestDepthwiseConvNPU_Padding)
create_test_channel_last_class(TestDepthwiseConvNPU2_Padding)

# test padding SAME
create_test_padding_SAME_class(TestDepthwiseConvNPU_Padding)
create_test_padding_SAME_class(TestDepthwiseConvNPU2_Padding)
create_test_padding_SAME_class(TestDepthwiseConvNPU3_Padding)

# test padding VALID
create_test_padding_VALID_class(TestDepthwiseConvNPU_Padding)
create_test_padding_VALID_class(TestDepthwiseConvNPU2_Padding)
create_test_padding_VALID_class(TestDepthwiseConvNPU3_Padding)

if __name__ == '__main__':
    unittest.main()
