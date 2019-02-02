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
from paddle.fluid.tests.unittests.test_mul_op import TestMulOp, TestMulOp2, TestFP16MulOp1, TestFP16MulOp2


class TestNGRAPHMulOp(TestMulOp):
    def setUp(self):
        super(TestNGRAPHMulOp, self).setUp()
        self._cpu_only = True

    def init_dtype_type(self):
        pass


class TestNGRAPHMulOp2(TestMulOp2):
    def setUp(self):
        super(TestNGRAPHMulOp2, self).setUp()
        self._cpu_only = True

    def init_dtype_type(self):
        pass


class TestNGRAPHFP16MulOp1(TestFP16MulOp1):
    def setUp(self):
        super(TestNGRAPHFP16MulOp1, self).setUp()
        self._cpu_only = True

    def init_dtype_type(self):
        pass


class TestNGRAPHFP16MulOp2(TestFP16MulOp2):
    def setUp(self):
        super(TestNGRAPHFP16MulOp2, self).setUp()
        self._cpu_only = True

    def init_dtype_type(self):
        pass


if __name__ == "__main__":
    unittest.main()
