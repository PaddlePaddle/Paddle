# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.fluid.tests.unittests.test_conv3d_op import TestConv3dOp, TestCase1, TestWithGroup1, TestWithGroup2, TestWith1x1, TestWithInput1x1Filter1x1


class TestMKLDNN(TestConv3dOp):
    def init_kernel_type(self):
        self.use_mkldnn = True
        self.data_format = "NCHW"


class TestMKLDNNCase1(TestCase1):
    def init_kernel_type(self):
        self.use_mkldnn = True
        self.data_format = "NCHW"


class TestMKLDNNGroup1(TestWithGroup1):
    def init_kernel_type(self):
        self.use_mkldnn = True
        self.data_format = "NCHW"


class TestMKLDNNGroup2(TestWithGroup2):
    def init_kernel_type(self):
        self.use_mkldnn = True
        self.data_format = "NCHW"


class TestMKLDNNWith1x1(TestWith1x1):
    def init_kernel_type(self):
        self.use_mkldnn = True
        self.data_format = "NCHW"


class TestMKLDNNWithInput1x1Filter1x1(TestWithInput1x1Filter1x1):
    def init_kernel_type(self):
        self.use_mkldnn = True
        self.data_format = "NCHW"


if __name__ == '__main__':
    unittest.main()
