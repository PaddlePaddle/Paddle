#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest


class TestSizeOp(OpTest):
    def setUp(self):
        self.op_type = "size"
        self.shape = []
        self.config()
        input = np.zeros(self.shape, dtype='bool')
        self.inputs = {'Input': input}
        self.outputs = {'Out': np.array([np.size(input)], dtype='int64')}

    def config(self):
        pass

    def test_check_output(self):
        self.check_output()


class TestRank1Tensor(TestSizeOp):
    def config(self):
        self.shape = [2]


class TestRank2Tensor(TestSizeOp):
    def config(self):
        self.shape = [2, 3]


class TestRank3Tensor(TestSizeOp):
    def config(self):
        self.shape = [2, 3, 100]


class TestLargeTensor(TestSizeOp):
    def config(self):
        self.shape = [2**10]


if __name__ == '__main__':
    unittest.main()
