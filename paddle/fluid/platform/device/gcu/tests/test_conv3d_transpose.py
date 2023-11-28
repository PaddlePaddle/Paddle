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


class TestConv3DTransposeOp(unittest.TestCase):
    def setUp(self):
        self.init_data_format()
        self.init_group()
        self.init_dilation()

        self.init_test_case()
        self.init_paddings()

        self.init_test_case_2()

    def test_api(self):
        test = ApiBase(
            func=paddle.nn.functional.conv3d_transpose,
            feed_names=['data', 'kernel'],
            feed_shapes=[self.input_size, self.filter_size],
            is_train=True,
        )
        np.random.seed(1)
        data = np.random.random(self.input_size).astype('float32')
        kernel = np.random.random(self.filter_size).astype('float32')
        attrs = {
            'stride': self.stride,
            'padding': self.pad,
            'dilation': self.dilations,
            'groups': self.groups,
            'data_format': self.data_format,
        }
        test.run(feed=[data, kernel], **attrs)

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
        self.padding_algorithm = "EXPLICIT"
        self.pad = [0, 0, 0]
        self.stride = [1, 1, 1]
        self.dilations = [1, 1, 1]
        self.groups = 1
        self.input_size = [2, 3, 5, 5, 5]  # NCDHW
        f_c = self.input_size[1]
        self.filter_size = [f_c, 6, 3, 3, 3]


class TestWithSymmetricPad(TestConv3DTransposeOp):
    def init_test_case(self):
        self.check_no_input = True
        self.pad = [1, 1, 1]
        self.stride = [1, 1, 1]
        self.dilations = [1, 1, 1]
        self.groups = 1
        self.input_size = [1, 2, 5, 5, 5]  # NCDHW
        f_c = self.input_size[1]
        self.filter_size = [f_c, 6, 3, 3, 3]


class TestWithSAMEPad(TestConv3DTransposeOp):
    def init_test_case(self):
        self.stride = [1, 1, 2]
        self.dilations = [1, 2, 1]
        self.groups = 1
        self.input_size = [1, 2, 5, 5, 6]  # NCDHW
        f_c = self.input_size[1]
        self.filter_size = [f_c, 6, 3, 3, 4]
        self.pad = [1, 1, 1]


class TestWithAsymmetricPad(TestConv3DTransposeOp):
    def init_test_case(self):
        self.pad = [1, 0, 1, 0, 1, 2]
        self.stride = [1, 1, 1]
        self.dilations = [1, 1, 1]
        self.groups = 1
        self.input_size = [1, 2, 5, 5, 5]  # NCDHW
        f_c = self.input_size[1]
        self.filter_size = [f_c, 6, 3, 3, 3]


class TestWithVALIDPad(TestConv3DTransposeOp):
    def init_test_case(self):
        self.stride = [2, 1, 1]
        self.dilations = [1, 1, 1]
        self.groups = 1
        self.input_size = [1, 2, 5, 5, 5]  # NCDHW
        f_c = self.input_size[1]
        self.filter_size = [f_c, 6, 3, 4, 3]
        self.pad = 'VALID'


class TestWithStride(TestConv3DTransposeOp):
    def init_test_case(self):
        self.check_no_filter = True
        self.pad = [1, 1, 1]
        self.stride = [2, 2, 2]
        self.dilations = [1, 1, 1]
        self.groups = 1
        self.input_size = [1, 2, 5, 5, 5]  # NCDHW
        f_c = self.input_size[1]
        self.filter_size = [f_c, 6, 3, 3, 3]


class TestWithDilation(TestConv3DTransposeOp):
    def init_test_case(self):
        self.pad = [1, 1, 1]
        self.stride = [1, 1, 1]
        self.dilations = [2, 2, 2]
        self.groups = 1
        self.input_size = [1, 2, 5, 5, 5]  # NCDHW
        f_c = self.input_size[1]
        self.filter_size = [f_c, 6, 3, 3, 3]


class Test_NHWC(TestConv3DTransposeOp):
    def init_test_case(self):
        self.pad = [0, 0, 0]
        self.stride = [1, 1, 1]
        self.dilations = [1, 1, 1]
        self.groups = 1
        self.input_size = [1, 5, 5, 5, 2]  # NDHWC
        f_c = self.input_size[-1]
        self.filter_size = [f_c, 6, 3, 3, 3]
        self.data_format = 'NDHWC'


def test_conv3d():

    z = TestWithSAMEPad()
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
