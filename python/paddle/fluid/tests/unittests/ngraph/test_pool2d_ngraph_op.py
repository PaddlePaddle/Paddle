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

from paddle.fluid.tests.unittests.test_pool2d_op import *


class TestNGRAPHPool2D_Op(TestPool2D_Op):
    def init_test_case(self):
        super(TestNGRAPHPool2D_Op, self).init_test_case()


class TestNGRAPHCase1(TestCase1):
    def init_test_case(self):
        super(TestNGRAPHCase1, self).init_test_case()


class TestNGRAPHCase2(TestCase2):
    def init_test_case(self):
        super(TestNGRAPHCase2, self).init_test_case()


class TestNGRAPHCase3(TestCase3):
    def init_pool_type(self):
        super(TestNGRAPHCase3, self).init_pool_type()


class TestNGRAPHCase4(TestCase4):
    def init_pool_type(self):
        super(TestNGRAPHCase4, self).init_pool_type()


class TestNGRAPHCase5(TestCase5):
    def init_pool_type(self):
        super(TestNGRAPHCase5, self).init_pool_type()


if __name__ == '__main__':
    unittest.main()
