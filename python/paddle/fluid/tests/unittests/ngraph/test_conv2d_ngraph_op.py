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
from paddle.fluid.tests.unittests.test_conv2d_op import *


class TestNGRAPH(TestConv2dOp):
    def init_kernel_type(self):
        super(TestNGRAPH, self).init_kernel_type()


class TestNGRAPHWithPad(TestWithPad):
    def init_kernel_type(self):
        super(TestNGRAPHWithPad, self).init_kernel_type()


class TestNGRAPHWithStride(TestWithStride):
    def init_kernel_type(self):
        super(TestNGRAPHWithStride, self).init_kernel_type()


class TestNGRAPHWithGroup(TestWithGroup):
    def init_kernel_type(self):
        super(TestNGRAPHWithGroup, self).init_kernel_type()


class TestNGRAPHWith1x1(TestWith1x1):
    def init_kernel_type(self):
        super(TestNGRAPHWith1x1, self).init_kernel_type()


class TestNGRAPHWithInput1x1Filter1x1(TestWithInput1x1Filter1x1):
    def init_kernel_type(self):
        super(TestNGRAPHWithInput1x1Filter1x1, self).init_kernel_type()


if __name__ == '__main__':
    unittest.main()
