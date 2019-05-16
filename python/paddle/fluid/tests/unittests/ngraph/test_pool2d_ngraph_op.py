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


class TestNGRAPHCeilMode(TestCase1):
    def setUp(self):
        super(TestNGRAPHCeilMode, self).setUp()

    def init_ceil_mode(self):
        self.ceil_mode = True


class TestNGRAPHAdaptive(TestCase1):
    def setUp(self):
        super(TestNGRAPHAdaptive, self).setUp()

    def init_adaptive(self):
        self.adaptive = True


if __name__ == '__main__':
    unittest.main()
