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

from __future__ import print_function

import unittest
from paddle.fluid.tests.unittests.test_pool2d_op import TestPool2D_Op, TestCase1, TestCase2, TestCase3, TestCase4, TestCase5


def create_test_mkldnn_use_ceil_class(parent):
    class TestMKLDNNPool2DUseCeilCase(parent):
        def init_kernel_type(self):
            self.use_mkldnn = True

        def init_ceil_mode(self):
            self.ceil_mode = True

    cls_name = "{0}_{1}".format(parent.__name__, "MKLDNNCeilModeCast")
    TestMKLDNNPool2DUseCeilCase.__name__ = cls_name
    globals()[cls_name] = TestMKLDNNPool2DUseCeilCase


create_test_mkldnn_use_ceil_class(TestPool2D_Op)
create_test_mkldnn_use_ceil_class(TestCase1)
create_test_mkldnn_use_ceil_class(TestCase2)


def create_test_mkldnn_class(parent):
    class TestMKLDNNCase(parent):
        def init_kernel_type(self):
            self.use_mkldnn = True

    cls_name = "{0}_{1}".format(parent.__name__, "MKLDNNOp")
    TestMKLDNNCase.__name__ = cls_name
    globals()[cls_name] = TestMKLDNNCase


create_test_mkldnn_class(TestPool2D_Op)
create_test_mkldnn_class(TestCase1)
create_test_mkldnn_class(TestCase2)
create_test_mkldnn_class(TestCase3)
create_test_mkldnn_class(TestCase4)
create_test_mkldnn_class(TestCase5)

if __name__ == '__main__':
    unittest.main()
