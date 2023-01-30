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
<<<<<<< HEAD
=======
from __future__ import print_function
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

import unittest
import numpy as np

import paddle
import paddle.fluid as fluid
import sys

sys.path.append("..")
from op_test import OpTest
from test_conv2d_op import conv2d_forward_naive
from paddle import ParamAttr
from paddle.regularizer import L2Decay
from paddle.nn.initializer import KaimingNormal

paddle.enable_static()
SEED = 2021


def create_test_channel_last_class(parent):
<<<<<<< HEAD
    class TestChannelLastCase(parent):
=======

    class TestChannelLastCase(parent):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def init_data_format(self):
            self.data_format = "NHWC"

        def init_test_case_2(self):
            N, C, H, W = self.input_size
            self.input_size = [N, H, W, C]

    cls_name = "{0}_{1}".format(parent.__name__, "ChannelLast")
    TestChannelLastCase.__name__ = cls_name
    globals()[cls_name] = TestChannelLastCase


def create_test_padding_SAME_class(parent):
<<<<<<< HEAD
    class TestPaddingSMAECase(parent):
=======

    class TestPaddingSMAECase(parent):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def init_paddings(self):
            self.pad = [0, 0]
            self.padding_algorithm = "SAME"

    cls_name = "{0}_{1}".format(parent.__name__, "PaddingSAMEOp")
    TestPaddingSMAECase.__name__ = cls_name
    globals()[cls_name] = TestPaddingSMAECase


def create_test_padding_VALID_class(parent):
<<<<<<< HEAD
    class TestPaddingVALIDCase(parent):
=======

    class TestPaddingVALIDCase(parent):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def init_paddings(self):
            self.pad = [1, 1]
            self.padding_algorithm = "VALID"

    cls_name = "{0}_{1}".format(parent.__name__, "PaddingVALIDOp")
    TestPaddingVALIDCase.__name__ = cls_name
    globals()[cls_name] = TestPaddingVALIDCase


def create_test_fp16_class(parent):
<<<<<<< HEAD
    class TestFp16Case(parent):
=======

    class TestFp16Case(parent):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def init_data_type(self):
            self.dtype = np.float16

    cls_name = "{0}_{1}".format(parent.__name__, "Fp16")
    TestFp16Case.__name__ = cls_name
    globals()[cls_name] = TestFp16Case


class TestDepthwiseConvNPU(OpTest):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.set_npu()
        self.op_type = "depthwise_conv2d"
        self.init_data_format()
        self.init_data_type()
        self.init_test_case()
        self.init_test_case_2()

        conv2d_param = {
            'stride': self.stride,
            'pad': self.pad,
<<<<<<< HEAD
            'dilation': self.dilations,
=======
            'dilation': self.dilations
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        }

        input = np.random.random(self.input_size).astype(self.dtype)
        filter = np.random.uniform(-1, 1, self.filter_size).astype(self.dtype)

<<<<<<< HEAD
        output, _, _, _, _ = conv2d_forward_naive(
            input,
            filter,
            self.groups,
            conv2d_param,
            "EXPLICIT",
            self.data_format,
        )
=======
        output, _, _, _, _ = conv2d_forward_naive(input, filter, self.groups,
                                                  conv2d_param, "EXPLICIT",
                                                  self.data_format)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        output = output.astype(self.dtype)

        self.inputs = {
            'Input': OpTest.np_dtype_to_fluid_dtype(input),
<<<<<<< HEAD
            'Filter': OpTest.np_dtype_to_fluid_dtype(filter),
=======
            'Filter': OpTest.np_dtype_to_fluid_dtype(filter)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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
        self.input_size = [2, 12, 5, 5]  # NCHW
        self.groups = 12
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [12, f_c, 3, 3]

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-2)

    def test_check_grad(self):
        if self.dilations[0] == 1 and self.dilations[1] == 1:
            if self.dtype == np.float16:
<<<<<<< HEAD
                self.check_grad_with_place(
                    self.place,
                    {'Input', 'Filter'},
                    'Output',
                    max_relative_error=0.9,
                )
            else:
                self.check_grad_with_place(
                    self.place,
                    {'Input', 'Filter'},
                    'Output',
                    max_relative_error=0.03,
                    numeric_place=paddle.CPUPlace(),
                )

    def test_check_grad_no_filter(self):
        if self.dtype == np.float16:
            self.check_grad_with_place(
                self.place,
                ['Input'],
                'Output',
                no_grad_set=set(['Filter']),
                max_relative_error=0.9,
            )
        else:
            self.check_grad_with_place(
                self.place,
                ['Input'],
                'Output',
                no_grad_set=set(['Filter']),
                max_relative_error=0.03,
                numeric_place=paddle.CPUPlace(),
            )
=======
                self.check_grad_with_place(self.place, {'Input', 'Filter'},
                                           'Output',
                                           max_relative_error=0.9)
            else:
                self.check_grad_with_place(self.place, {'Input', 'Filter'},
                                           'Output',
                                           max_relative_error=0.03,
                                           numeric_place=paddle.CPUPlace())

    def test_check_grad_no_filter(self):
        if self.dtype == np.float16:
            self.check_grad_with_place(self.place, ['Input'],
                                       'Output',
                                       no_grad_set=set(['Filter']),
                                       max_relative_error=0.9)
        else:
            self.check_grad_with_place(self.place, ['Input'],
                                       'Output',
                                       no_grad_set=set(['Filter']),
                                       max_relative_error=0.03,
                                       numeric_place=paddle.CPUPlace())
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def test_check_grad_no_input(self):
        if self.dilations[0] == 1 and self.dilations[1] == 1:
            if self.dtype == np.float16:
<<<<<<< HEAD
                self.check_grad_with_place(
                    self.place,
                    ['Filter'],
                    'Output',
                    no_grad_set=set(['Input']),
                    max_relative_error=0.9,
                )
            else:
                self.check_grad_with_place(
                    self.place,
                    ['Filter'],
                    'Output',
                    no_grad_set=set(['Input']),
                    max_relative_error=0.03,
                    numeric_place=paddle.CPUPlace(),
                )
=======
                self.check_grad_with_place(self.place, ['Filter'],
                                           'Output',
                                           no_grad_set=set(['Input']),
                                           max_relative_error=0.9)
            else:
                self.check_grad_with_place(self.place, ['Filter'],
                                           'Output',
                                           no_grad_set=set(['Input']),
                                           max_relative_error=0.03,
                                           numeric_place=paddle.CPUPlace())
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def init_data_format(self):
        self.data_format = "NCHW"

    def init_data_type(self):
        self.dtype = np.float32

    def init_test_case_2(self):
        pass


class TestDepthwiseConvNPU2(TestDepthwiseConvNPU):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_test_case(self):
        self.pad = [1, 1]
        self.dilations = [1, 1]
        self.stride = [1, 1]
        self.input_size = [2, 12, 5, 5]  # NCHW
        self.groups = 12
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [12, f_c, 3, 3]


class TestDepthwiseConvNPU3(TestDepthwiseConvNPU):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_test_case(self):
        self.pad = [1, 1]
        self.dilations = [1, 1]
        self.stride = [2, 2]
        self.input_size = [2, 12, 5, 5]  # NCHW
        self.groups = 12
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [12, f_c, 3, 3]


class TestDepthwiseConvNPU4(TestDepthwiseConvNPU):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_test_case(self):
        self.pad = [1, 1]
        self.dilations = [1, 1]
        self.stride = [1, 1]
        self.input_size = [2, 12, 5, 5]  # NCHW
        self.groups = 12
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [12, f_c, 3, 3]


class TestDepthwiseConvNPU_Padding(OpTest):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.op_type = "depthwise_conv2d"
        self.dtype = np.float32
        self.set_npu()
        self.init_data_format()
        self.init_data_type()
        self.init_paddings()
        self.init_test_case()
        self.init_test_case_2()

        conv2d_param = {
            'stride': self.stride,
            'pad': self.pad,
<<<<<<< HEAD
            'dilation': self.dilations,
=======
            'dilation': self.dilations
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        }

        input = np.random.random(self.input_size).astype(self.dtype)
        filter = np.random.uniform(-1, 1, self.filter_size).astype(self.dtype)

<<<<<<< HEAD
        output, _, _, _, _ = conv2d_forward_naive(
            input,
            filter,
            self.groups,
            conv2d_param,
            self.padding_algorithm,
            self.data_format,
        )
=======
        output, _, _, _, _ = conv2d_forward_naive(input, filter, self.groups,
                                                  conv2d_param,
                                                  self.padding_algorithm,
                                                  self.data_format)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        output = output.astype(self.dtype)

        self.inputs = {
            'Input': OpTest.np_dtype_to_fluid_dtype(input),
<<<<<<< HEAD
            'Filter': OpTest.np_dtype_to_fluid_dtype(filter),
=======
            'Filter': OpTest.np_dtype_to_fluid_dtype(filter)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        }

        self.attrs = {
            'strides': self.stride,
            'paddings': self.pad,
            'padding_algorithm': self.padding_algorithm,
            'groups': self.groups,
            'dilations': self.dilations,
<<<<<<< HEAD
            'data_format': self.data_format,
=======
            'data_format': self.data_format
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        }
        self.outputs = {'Output': output}

    def set_npu(self):
        self.__class__.use_npu = True
        self.place = paddle.NPUPlace(0)

    def init_test_case(self):
        self.pad = [1, 1, 0, 1]
        self.dilations = [1, 1]
        self.stride = [2, 2]
        self.input_size = [2, 12, 5, 5]  # NCHW
        self.groups = 12
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [12, f_c, 3, 3]

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-2)

    def test_check_grad(self):
        if self.dtype == np.float16:
<<<<<<< HEAD
            self.check_grad_with_place(
                self.place,
                {'Input', 'Filter'},
                'Output',
                max_relative_error=1.2,
            )
        else:
            self.check_grad_with_place(
                self.place,
                {'Input', 'Filter'},
                'Output',
                max_relative_error=0.03,
                numeric_place=paddle.CPUPlace(),
            )

    def test_check_grad_no_filter(self):
        if self.dtype == np.float16:
            self.check_grad_with_place(
                self.place,
                ['Input'],
                'Output',
                max_relative_error=0.7,
                no_grad_set=set(['Filter']),
            )
        else:
            self.check_grad_with_place(
                self.place,
                ['Input'],
                'Output',
                max_relative_error=0.03,
                no_grad_set=set(['Filter']),
                numeric_place=paddle.CPUPlace(),
            )

    def test_check_grad_no_input(self):
        if self.dtype == np.float16:
            self.check_grad_with_place(
                self.place,
                ['Filter'],
                'Output',
                max_relative_error=0.8,
                no_grad_set=set(['Input']),
            )
        else:
            self.check_grad_with_place(
                self.place,
                ['Filter'],
                'Output',
                max_relative_error=0.03,
                no_grad_set=set(['Input']),
                numeric_place=paddle.CPUPlace(),
            )
=======
            self.check_grad_with_place(self.place, {'Input', 'Filter'},
                                       'Output',
                                       max_relative_error=1.2)
        else:
            self.check_grad_with_place(self.place, {'Input', 'Filter'},
                                       'Output',
                                       max_relative_error=0.03,
                                       numeric_place=paddle.CPUPlace())

    def test_check_grad_no_filter(self):
        if self.dtype == np.float16:
            self.check_grad_with_place(self.place, ['Input'],
                                       'Output',
                                       max_relative_error=0.7,
                                       no_grad_set=set(['Filter']))
        else:
            self.check_grad_with_place(self.place, ['Input'],
                                       'Output',
                                       max_relative_error=0.03,
                                       no_grad_set=set(['Filter']),
                                       numeric_place=paddle.CPUPlace())

    def test_check_grad_no_input(self):
        if self.dtype == np.float16:
            self.check_grad_with_place(self.place, ['Filter'],
                                       'Output',
                                       max_relative_error=0.8,
                                       no_grad_set=set(['Input']))
        else:
            self.check_grad_with_place(self.place, ['Filter'],
                                       'Output',
                                       max_relative_error=0.03,
                                       no_grad_set=set(['Input']),
                                       numeric_place=paddle.CPUPlace())
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def init_data_format(self):
        self.data_format = "NCHW"

    def init_data_type(self):
        self.dtype = np.float32

    def init_paddings(self):
        self.pad = [1, 1, 0, 1]
        self.padding_algorithm = "EXPLICIT"

    def init_test_case_2(self):
        pass


class TestDepthwiseConvNPU2_Padding(TestDepthwiseConvNPU_Padding):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_test_case(self):
        self.pad = [1, 1, 0, 1]
        self.dilations = [1, 1]
        self.stride = [1, 1]
        self.input_size = [2, 12, 5, 5]  # NCHW
        self.groups = 12
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [12, f_c, 3, 3]

    def init_paddings(self):
        self.pad = [0, 1, 0, 2]
        self.padding_algorithm = "EXPLICIT"


class TestDepthwiseConvNPU3_Padding(TestDepthwiseConvNPU_Padding):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_test_case(self):
        self.pad = [1, 1, 0, 1]
        self.dilations = [1, 1]
        self.stride = [2, 2]
        self.input_size = [2, 12, 5, 5]  # NCHW
        self.groups = 12
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [12, f_c, 3, 3]

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

create_test_fp16_class(TestDepthwiseConvNPU)
create_test_fp16_class(TestDepthwiseConvNPU2)
create_test_fp16_class(TestDepthwiseConvNPU_Padding)
create_test_fp16_class(TestDepthwiseConvNPU2_Padding)
create_test_fp16_class(TestDepthwiseConvNPU3_Padding)

if __name__ == '__main__':
    unittest.main()
