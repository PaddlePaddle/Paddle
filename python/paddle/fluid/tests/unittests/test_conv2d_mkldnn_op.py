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

from test_conv2d_op import TestConv2dOp, TestWithPad, TestWithStride


class TestMKLDNN(TestConv2dOp):
    def init_kernel_type(self):
        self.use_mkldnn = True
        self.data_format = "NCHW"


class TestMKLDNNWithPad(TestWithPad):
    def init_kernel_type(self):
        self.use_mkldnn = True
        self.data_format = "NCHW"


class TestMKLDNNWithStride(TestWithStride):
    def init_kernel_type(self):
        self.use_mkldnn = True
        self.data_format = "NCHW"


if __name__ == '__main__':
    unittest.main()
