# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from api_base import ApiBase

import paddle


class TestConv3DOp(unittest.TestCase):
    def setUp(self):
        self.init_data_format()
        self.init_group()
        self.init_dilation()

        self.init_test_case()
        self.init_paddings()

        self.init_test_case_2()

    def test_api(self):
        test = ApiBase(
            func=paddle.nn.functional.conv3d,
            feed_names=['data', 'kernel'],
            feed_shapes=[self.input_size, self.filter_size],
            is_train=True,
        )
        np.random.seed(1)
        data = np.random.uniform(-1, 1, self.input_size).astype('float32')
        kernel = np.random.uniform(-1, 1, self.filter_size).astype('float32')
        test.run(
            feed=[data, kernel],
            bias=None,
            stride=self.stride,
            dilation=self.dilations,
            data_format=self.data_format,
            padding=self.pad,
            groups=self.groups,
        )

    def init_test_case_2(self):
        pass

    def init_paddings(self):
        pass

    def init_data_format(self):
        self.data_format = "NCDHW"

    def init_group(self):
        self.groups = 1

    def init_dilation(self):
        self.dilations = [1, 1, 1]

    def init_test_case(self):
        self.pad = [1, 1, 1]
        self.stride = [1, 1, 1]
        self.input_size = [2, 3, 4, 4, 4]  # NCDHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 3, 3]


class TestWith1x1(TestConv3DOp):
    def init_test_case(self):
        self.pad = [0, 0, 0]
        self.stride = [1, 1, 1]
        self.input_size = [2, 3, 4, 4, 4]
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [120, f_c, 1, 1, 1]


class TestWithInput1x1Filter1x1(TestConv3DOp):
    def init_test_case(self):
        self.pad = [0, 0, 0]
        self.stride = [1, 1, 1]
        self.input_size = [40, 3, 1, 1, 1]
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [120, f_c, 1, 1, 1]

    def init_dilation(self):
        self.dilations = [1, 1, 1]


class TestWithDilation(TestConv3DOp):
    def init_test_case(self):
        self.pad = [0, 0, 0]
        self.stride = [1, 1, 1]
        self.input_size = [2, 3, 6, 6, 6]
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [24, f_c, 2, 2, 2]

    def init_dilation(self):
        self.dilations = [2, 2, 2]


class TestConv3DOp_AsyPadding(TestConv3DOp):
    def init_test_case(self):
        self.stride = [1, 1, 2]
        self.input_size = [2, 3, 4, 4, 4]  # NCDHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 3, 3]
        self.pad = [1, 0, 1, 0, 0, 2]
        self.padding_algorithm = "EXPLICIT"


class TestConv3DOp_DiffDataInDiffDim(TestConv3DOp):
    def init_test_case(self):
        self.stride = [1, 1, 2]
        self.input_size = [2, 3, 4, 5, 5]  # NCDHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 4, 3]
        self.pad = [1, 0, 1, 0, 0, 2]
        self.padding_algorithm = "EXPLICIT"


def create_test_padding_SAME_class(parent):
    class TestPaddingSMAECase(parent):
        def init_paddings(self):

            self.pad = "SAME"

    cls_name = "{}_{}".format(parent.__name__, "PaddingSAMEOp")
    TestPaddingSMAECase.__name__ = cls_name
    globals()[cls_name] = TestPaddingSMAECase


def create_test_padding_VALID_class(parent):
    class TestPaddingVALIDCase(parent):
        def init_paddings(self):

            self.pad = "VALID"

    cls_name = "{}_{}".format(parent.__name__, "PaddingVALIDOp")
    TestPaddingVALIDCase.__name__ = cls_name
    globals()[cls_name] = TestPaddingVALIDCase


def create_test_channel_last_class(parent):
    class TestChannelLastCase(parent):
        def init_data_format(self):
            self.data_format = "NDHWC"

        def init_test_case_2(self):
            N, C, D, H, W = self.input_size
            self.input_size = [N, D, H, W, C]

    cls_name = "{}_{}".format(parent.__name__, "ChannelLast")
    TestChannelLastCase.__name__ = cls_name
    globals()[cls_name] = TestChannelLastCase


create_test_padding_SAME_class(TestConv3DOp_DiffDataInDiffDim)
create_test_padding_VALID_class(TestConv3DOp_DiffDataInDiffDim)
create_test_channel_last_class(TestConv3DOp_DiffDataInDiffDim)


def test_conv3d():
    class ss(TestConv3DOp):
        def init_test_case(self):
            self.pad = [0, 0, 0]
            self.stride = [1, 1, 1]
            self.input_size = [1, 2, 8, 8, 8]
            f_c = self.input_size[1]
            self.filter_size = [32, f_c, 3, 3, 3]
            self.data_format = "NDHWC"
            N, C, D, H, W = self.input_size
            self.input_size = [N, D, H, W, C]

    z = ss()
    z.setUp()
    z.test_api()
    # TestConv3DOp_AsyPadding_same()
    # TestConv3DOp_DiffDataInDiffDim_same()
    # TestConv3DOp()
    # TestWithGroup1()
    # TestWith1x1()
    # TestWithInput1x1Filter1x1()


if __name__ == "__main__":
    test_conv3d()
