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
import numpy as np
from paddle.fluid.tests.unittests.test_fill_constant_op import TestFillConstantOp1, TestFillConstantOp2, TestFillConstantOpWithSelectedRows


class TestNGRAPHFillConstantFP64(TestFillConstantOp1):
    def setUp(self):
        super(TestNGRAPHFillConstantFP64, self).setUp()

        self.attrs = {'shape': [123, 92], 'value': 3.8, 'dtype': 6}
        self.outputs = {'Out': np.full((123, 92), 3.8)}


class TestNGRAPHFillConstantINT32(TestFillConstantOp2):
    def setUp(self):
        super(TestNGRAPHFillConstantINT32, self).setUp()

        self.attrs = {'shape': [123, 92], 'dtype': 2}
        self.outputs = {'Out': np.full((123, 92), 0)}


class TestNGRAPHFillConstantINT64(TestFillConstantOp2):
    def setUp(self):
        super(TestNGRAPHFillConstantINT64, self).setUp()

        self.attrs = {'shape': [123, 92], 'dtype': 3}
        self.outputs = {'Out': np.full((123, 92), 0)}


if __name__ == "__main__":
    unittest.main()
