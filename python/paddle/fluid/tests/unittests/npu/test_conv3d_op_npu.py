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

import sys

sys.path.append("..")
import paddle
import paddle.fluid.core as core
from op_test import OpTest
import paddle.fluid as fluid

from test_conv3d_op import conv3d_forward_naive

paddle.enable_static()


def create_test_padding_SAME_class(parent):

    class TestPaddingSMAECase(parent):

        def init_paddings(self):
            self.pad = [0, 0, 0]
            self.padding_algorithm = "SAME"

    cls_name = "{0}_{1}".format(parent.__name__, "PaddingSAMEOp")
    TestPaddingSMAECase.__name__ = cls_name
    globals()[cls_name] = TestPaddingSMAECase


def create_test_padding_VALID_class(parent):

    class TestPaddingVALIDCase(parent):

        def init_paddings(self):
            self.pad = [1, 1, 1]
            self.padding_algorithm = "VALID"

    cls_name = "{0}_{1}".format(parent.__name__, "PaddingVALIDOp")
    TestPaddingVALIDCase.__name__ = cls_name
    globals()[cls_name] = TestPaddingVALIDCase


def create_test_channel_last_class(parent):

    class TestChannelLastCase(parent):

        def init_data_format(self):
            self.data_format = "NDHWC"

        def init_test_case_2(self):
            N, C, D, H, W = self.input_size
            self.input_size = [N, D, H, W, C]

    cls_name = "{0}_{1}".format(parent.__name__, "ChannelLast")
    TestChannelLastCase.__name__ = cls_name
    globals()[cls_name] = TestChannelLastCase


def create_test_fp16_class(parent):

    class TestFp16Case(parent):

        def init_dtype(self):
            self.dtype = np.float16

    cls_name = "{0}_{1}".format(parent.__name__, "Fp16")
    TestFp16Case.__name__ = cls_name
    globals()[cls_name] = TestFp16Case


class TestConv3DOp(OpTest):

    def setUp(self):
        self.op_type = "conv3d"
        self.set_npu()
        self.init_dtype()
        self.init_data_format()
        self.init_group()
        self.init_dilation()
        self.init_test_case()

        conv3d_param = {
            'stride': self.stride,
            'pad': self.pad,
            'dilations': self.dilations
        }

        input = np.random.random(self.input_size).astype(self.dtype)
        filter = np.random.random(self.filter_size).astype(self.dtype)
        output = conv3d_forward_naive(
            input,
            filter,
            self.groups,
            conv3d_param,
        ).astype(self.dtype)

        self.inputs = {
            'Input': OpTest.np_dtype_to_fluid_dtype(input),
            'Filter': OpTest.np_dtype_to_fluid_dtype(filter)
        }
        self.attrs = {
            'strides': self.stride,
            'paddings': self.pad,
            'groups': self.groups,
            'dilations': self.dilations,
            'data_format': self.data_format
        }
        self.outputs = {'Output': output}

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-2)

    def test_check_grad(self):
        if self.dtype == np.float16:
            return

        self.check_grad_with_place(self.place, {'Input', 'Filter'},
                                   'Output',
                                   max_relative_error=0.03,
                                   numeric_place=paddle.CPUPlace())

    def test_check_grad_no_filter(self):
        if self.dtype == np.float16:
            return

        self.check_grad_with_place(self.place, ['Input'],
                                   'Output',
                                   max_relative_error=0.03,
                                   no_grad_set=set(['Filter']),
                                   numeric_place=paddle.CPUPlace())

    def test_check_grad_no_input(self):
        if self.dtype == np.float16:
            return

        self.check_grad_with_place(self.place, ['Filter'],
                                   'Output',
                                   max_relative_error=0.03,
                                   no_grad_set=set(['Input']),
                                   numeric_place=paddle.CPUPlace())

    def set_npu(self):
        self.__class__.use_npu = True
        self.place = fluid.NPUPlace(0)

    def init_dtype(self):
        self.dtype = np.float32

    def init_data_format(self):
        self.data_format = "NCDHW"

    def init_group(self):
        self.groups = 1

    def init_dilation(self):
        self.dilations = [1, 1, 1]

    def init_test_case(self):
        self.pad = [0, 0, 0]
        self.stride = [1, 1, 1]
        self.input_size = [2, 3, 4, 4, 4]  # NCDHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 3, 3]


class TestCase1(TestConv3DOp):

    def init_test_case(self):
        self.pad = [1, 1, 1]
        self.stride = [1, 1, 1]
        self.input_size = [2, 3, 4, 4, 4]  # NCDHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 3, 3]


# ---- test asymmetric padding ----


class TestConv3DOp_2(OpTest):

    def setUp(self):
        self.op_type = "conv3d"
        self.set_npu()
        self.init_dtype()
        self.init_data_format()
        self.init_group()
        self.init_dilation()
        self.init_paddings()
        self.init_test_case()

        self.init_test_case_2()

        conv3d_param = {
            'stride': self.stride,
            'pad': self.pad,
            'dilations': self.dilations
        }

        input = np.random.random(self.input_size).astype(self.dtype)
        filter = np.random.random(self.filter_size).astype(self.dtype)
        output = conv3d_forward_naive(input, filter, self.groups, conv3d_param,
                                      self.padding_algorithm,
                                      self.data_format).astype(self.dtype)

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

    def test_check_output(self):
        self.check_output_with_place(paddle.NPUPlace(0), atol=1e-2)

    def test_check_grad(self):
        if self.dtype == np.float16:
            return

        self.check_grad_with_place(self.place, {'Input', 'Filter'},
                                   'Output',
                                   max_relative_error=0.03,
                                   numeric_place=paddle.CPUPlace())

    def test_check_grad_no_filter(self):
        if self.dtype == np.float16:
            return

        self.check_grad_with_place(self.place, ['Input'],
                                   'Output',
                                   max_relative_error=0.03,
                                   no_grad_set=set(['Filter']),
                                   numeric_place=paddle.CPUPlace())

    def test_check_grad_no_input(self):
        if self.dtype == np.float16:
            return

        self.check_grad_with_place(self.place, ['Filter'],
                                   'Output',
                                   max_relative_error=0.03,
                                   no_grad_set=set(['Input']),
                                   numeric_place=paddle.CPUPlace())

    def set_npu(self):
        self.__class__.use_npu = True
        self.place = fluid.NPUPlace(0)

    def init_dtype(self):
        self.dtype = np.float32

    def init_data_format(self):
        self.data_format = "NCDHW"

    def init_group(self):
        self.groups = 1

    def init_dilation(self):
        self.dilations = [1, 1, 1]

    def init_paddings(self):
        self.pad = [0, 0, 0]
        self.padding_algorithm = "EXPLICIT"

    def init_test_case(self):
        self.stride = [1, 1, 1]
        self.input_size = [2, 3, 4, 4, 4]  # NCDHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 3, 3]

    def init_test_case_2(self):
        pass


class TestConv3DOp_AsyPadding(TestConv3DOp_2):

    def init_test_case(self):
        self.stride = [1, 1, 2]
        self.input_size = [2, 3, 4, 4, 4]  # NCDHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 3, 3]

    def init_paddings(self):
        self.pad = [1, 0, 1, 0, 0, 2]
        self.padding_algorithm = "EXPLICIT"


class TestConv3DOp_DiffDataInDiffDim(TestConv3DOp_2):

    def init_test_case(self):
        self.stride = [1, 1, 2]
        self.input_size = [2, 3, 4, 5, 5]  # NCDHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 4, 3]

    def init_paddings(self):
        self.pad = [1, 0, 1, 0, 0, 2]
        self.padding_algorithm = "EXPLICIT"


class TestCase1_AsyPadding(TestConv3DOp_2):

    def init_test_case(self):
        self.stride = [1, 1, 1]
        self.input_size = [2, 3, 4, 4, 4]  # NCDHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 3, 3]

    def init_paddings(self):
        self.pad = [0, 0, 1, 0, 0, 2]
        self.padding_algorithm = "EXPLICIT"


# --------- test python API ---------------
class TestConv3DAPI(unittest.TestCase):

    def test_api(self):

        input_NDHWC = fluid.layers.data(name="input_NDHWC",
                                        shape=[2, 5, 5, 5, 3],
                                        append_batch_size=False,
                                        dtype="float32")

        input_NCDHW = fluid.layers.data(name="input_NCDHW",
                                        shape=[2, 3, 5, 5, 3],
                                        append_batch_size=False,
                                        dtype="float32")

        fluid.layers.conv3d(input=input_NDHWC,
                            num_filters=3,
                            filter_size=[3, 3, 3],
                            stride=[1, 1, 1],
                            padding=0,
                            dilation=[1, 1, 1],
                            groups=1,
                            data_format="NCDHW")

        fluid.layers.conv3d(input=input_NCDHW,
                            num_filters=3,
                            filter_size=[3, 3, 3],
                            stride=[1, 1, 1],
                            padding=[1, 2, 1, 0, 1, 0],
                            dilation=[1, 1, 1],
                            groups=1,
                            data_format="NCDHW")

        fluid.layers.conv3d(input=input_NCDHW,
                            num_filters=3,
                            filter_size=[3, 3, 3],
                            stride=[1, 1, 1],
                            padding=[[0, 0], [0, 0], [1, 1], [1, 1], [1, 1]],
                            dilation=[1, 1, 1],
                            groups=1,
                            data_format="NCDHW")

        fluid.layers.conv3d(input=input_NDHWC,
                            num_filters=3,
                            filter_size=[3, 3, 3],
                            stride=[1, 1, 1],
                            padding=[[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]],
                            dilation=[1, 1, 1],
                            groups=1,
                            data_format="NDHWC")

        fluid.layers.conv3d(input=input_NCDHW,
                            num_filters=3,
                            filter_size=[3, 3, 3],
                            stride=[1, 1, 1],
                            padding="SAME",
                            dilation=[1, 1, 1],
                            groups=1,
                            data_format="NCDHW")

        fluid.layers.conv3d(input=input_NCDHW,
                            num_filters=3,
                            filter_size=[3, 3, 3],
                            stride=[1, 1, 1],
                            padding="VALID",
                            dilation=[1, 1, 1],
                            groups=1,
                            data_format="NCDHW")


class TestConv3DAPI_Error(unittest.TestCase):

    def test_api(self):
        input = fluid.layers.data(name="input",
                                  shape=[2, 5, 5, 5, 4],
                                  append_batch_size=False,
                                  dtype="float32")

        # ValueError: cudnn
        def run_1():
            fluid.layers.conv3d(input=input,
                                num_filters=3,
                                filter_size=3,
                                stride=1,
                                padding=0,
                                dilation=1,
                                groups=1,
                                use_cudnn=[0],
                                data_format="NCDHW")

        self.assertRaises(ValueError, run_1)

        # ValueError: data_format
        def run_2():
            fluid.layers.conv3d(input=input,
                                num_filters=3,
                                filter_size=[3, 3, 3],
                                stride=[1, 1, 1],
                                padding=0,
                                dilation=[1, 1, 1],
                                groups=1,
                                use_cudnn=False,
                                data_format="NCHWC")

        self.assertRaises(ValueError, run_2)

        # ValueError: padding
        def run_3():
            fluid.layers.conv3d(input=input,
                                num_filters=3,
                                filter_size=3,
                                stride=1,
                                padding="SAMEE",
                                dilation=1,
                                groups=1,
                                use_cudnn=False,
                                data_format="NCDHW")

        self.assertRaises(ValueError, run_3)

        def run_4():
            fluid.layers.conv3d(input=input,
                                num_filters=3,
                                filter_size=3,
                                stride=1,
                                padding=[[0, 1], [0, 0], [0, 1], [0, 1], [0,
                                                                          1]],
                                dilation=1,
                                groups=1,
                                use_cudnn=False,
                                data_format="NCDHW")

        self.assertRaises(ValueError, run_4)

        def run_5():
            fluid.layers.conv3d(input=input,
                                num_filters=3,
                                filter_size=0,
                                stride=0,
                                padding=[[0, 1], [0, 1], [0, 1], [0, 1], [0,
                                                                          1]],
                                dilation=1,
                                groups=1,
                                use_cudnn=False,
                                data_format="NDHWC")

        self.assertRaises(ValueError, run_5)

        # ValueError: channel dimmention
        x = fluid.layers.data(name="x",
                              shape=[2, 5, 5, 5, -1],
                              append_batch_size=False,
                              dtype="float32")

        def run_6():
            fluid.layers.conv3d(input=x,
                                num_filters=3,
                                filter_size=3,
                                stride=1,
                                padding=0,
                                dilation=1,
                                groups=1,
                                use_cudnn=False,
                                data_format="NDHWC")

        self.assertRaises(ValueError, run_6)

        # ValueError: groups
        def run_7():
            fluid.layers.conv3d(input=input,
                                num_filters=3,
                                filter_size=3,
                                stride=1,
                                padding=0,
                                dilation=1,
                                groups=3,
                                use_cudnn=False,
                                data_format="NDHWC")

        self.assertRaises(ValueError, run_7)

        # ValueError: filter num
        def run_8():
            fluid.layers.conv3d(input=input,
                                num_filters=0,
                                filter_size=0,
                                stride=0,
                                padding=0,
                                dilation=0,
                                groups=1,
                                use_cudnn=False,
                                data_format="NDHWC")

        self.assertRaises(ValueError, run_8)


if __name__ == '__main__':
    unittest.main()
