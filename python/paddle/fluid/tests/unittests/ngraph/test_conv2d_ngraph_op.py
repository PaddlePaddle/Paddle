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

import unittest, sys
sys.path.append("../")
from test_conv2d_op import TestConv2dOp, TestWithPad, TestWithStride, TestWithGroup, TestWith1x1, TestWithInput1x1Filter1x1, TestDepthwiseConv, TestDepthwiseConv2, TestDepthwiseConv3, TestDepthwiseConvWithDilation, TestDepthwiseConvWithDilation2
import numpy as np


class TestNGRAPHWithStride(TestWithStride):
    def init_test_case(self):
        super(TestNGRAPHWithStride, self).init_test_case()
        self.use_cuda = False
        self.dtype = np.float32


class TestNGRAPHDepthwiseConv(TestDepthwiseConv):
    def init_test_case(self):
        super(TestNGRAPHDepthwiseConv, self).init_test_case()
        self.use_cuda = False
        self.dtype = np.float32


class TestNGRAPHDepthwiseConv2(TestDepthwiseConv2):
    def init_test_case(self):
        super(TestNGRAPHDepthwiseConv2, self).init_test_case()
        self.use_cuda = False
        self.dtype = np.float32


class TestNGRAPHDepthwiseConv3(TestDepthwiseConv3):
    def init_test_case(self):
        super(TestNGRAPHDepthwiseConv3, self).init_test_case()
        self.use_cuda = False
        self.dtype = np.float32


class TestNGRAPHDepthwiseConvWithDilation(TestDepthwiseConvWithDilation):
    def init_test_case(self):
        super(TestNGRAPHDepthwiseConvWithDilation, self).init_test_case()
        self.use_cuda = False
        self.dtype = np.float32


class TestNGRAPHDepthwiseConvWithDilation2(TestDepthwiseConvWithDilation2):
    def init_test_case(self):
        super(TestNGRAPHDepthwiseConvWithDilation2, self).init_test_case()
        self.use_cuda = False
        self.dtype = np.float32


del TestWithStride, TestDepthwiseConv, TestDepthwiseConv2, TestDepthwiseConv3, TestDepthwiseConvWithDilation, TestDepthwiseConvWithDilation2

if __name__ == '__main__':
    unittest.main()
