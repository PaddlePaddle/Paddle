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

import unittest

import numpy as np
from test_conv3d_op import (
    TestCase1,
    TestConv3DOp,
    TestWith1x1,
    TestWithGroup1,
    TestWithGroup2,
    TestWithInput1x1Filter1x1,
)


class TestMKLDNN(TestConv3DOp):
    def init_kernel_type(self):
        self.use_mkldnn = True
        self.data_format = "NCHW"
        self.dtype = np.float32
        self.check_pir_onednn = True


class TestMKLDNNCase1(TestCase1):
    def init_kernel_type(self):
        self.use_mkldnn = True
        self.data_format = "NCHW"
        self.dtype = np.float32
        self.check_pir_onednn = True


class TestMKLDNNGroup1(TestWithGroup1):
    def init_kernel_type(self):
        self.use_mkldnn = True
        self.data_format = "NCHW"
        self.dtype = np.float32
        self.check_pir_onednn = True


class TestMKLDNNGroup2(TestWithGroup2):
    def init_kernel_type(self):
        self.use_mkldnn = True
        self.data_format = "NCHW"
        self.dtype = np.float32
        self.check_pir_onednn = True


class TestMKLDNNWith1x1(TestWith1x1):
    def init_kernel_type(self):
        self.use_mkldnn = True
        self.data_format = "NCHW"
        self.dtype = np.float32
        self.check_pir_onednn = True


class TestMKLDNNWithInput1x1Filter1x1(TestWithInput1x1Filter1x1):
    def init_kernel_type(self):
        self.use_mkldnn = True
        self.data_format = "NCHW"
        self.dtype = np.float32
        self.check_pir_onednn = True


class TestConv3DOp_AsyPadding_MKLDNN(TestConv3DOp):
    def init_kernel_type(self):
        self.use_mkldnn = True
        self.data_format = "NCHW"
        self.dtype = np.float32
        self.check_pir_onednn = True

    def init_paddings(self):
        self.pad = [1, 0, 1, 0, 0, 2]
        self.padding_algorithm = "EXPLICIT"


class TestConv3DOp_Same_MKLDNN(TestConv3DOp_AsyPadding_MKLDNN):
    def init_paddings(self):
        self.pad = [0, 0, 0]
        self.padding_algorithm = "SAME"

    def init_kernel_type(self):
        self.use_mkldnn = True
        self.data_format = "NCHW"
        self.dtype = np.float32
        self.check_pir_onednn = True


class TestConv3DOp_Valid_MKLDNN(TestConv3DOp_AsyPadding_MKLDNN):
    def init_paddings(self):
        self.pad = [1, 1, 1]
        self.padding_algorithm = "VALID"

    def init_kernel_type(self):
        self.use_mkldnn = True
        self.data_format = "NCHW"
        self.dtype = np.float32
        self.check_pir_onednn = True


if __name__ == '__main__':
    from paddle import enable_static

    enable_static()
    unittest.main()
