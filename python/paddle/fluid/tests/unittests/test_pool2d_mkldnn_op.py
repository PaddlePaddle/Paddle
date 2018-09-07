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
from test_pool2d_op import TestPool2d_Op, TestCase1, TestCase2, TestCase3, TestCase4, TestCase5


class TestMKLDNNCase1(TestPool2d_Op):
    def init_kernel_type(self):
        self.use_mkldnn = True


class TestMKLDNNCase2(TestCase1):
    def init_kernel_type(self):
        self.use_mkldnn = True


class TestMKLDNNCase3(TestCase2):
    def init_kernel_type(self):
        self.use_mkldnn = True


class TestMKLDNNCase4(TestCase3):
    def init_kernel_type(self):
        self.use_mkldnn = True


class TestMKLDNNCase5(TestCase4):
    def init_kernel_type(self):
        self.use_mkldnn = True


class TestMKLDNNCase6(TestCase5):
    def init_kernel_type(self):
        self.use_mkldnn = True


if __name__ == '__main__':
    unittest.main()
