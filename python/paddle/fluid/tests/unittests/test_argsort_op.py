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
import numpy as np
from op_test import OpTest


class TestArgsortOp(OpTest):
    def setUp(self):
        self.init_axis()
        x = np.random.random((2, 3, 4, 5, 10)).astype("float32")
        if self.axis < 0:
            self.axis = self.axis + len(x.shape)
        self.indices = np.argsort(x, kind='quicksort', axis=self.axis)
        self.out = np.sort(x, kind='quicksort', axis=self.axis)
        self.op_type = "argsort"
        self.inputs = {'X': x}
        self.attrs = {'axis': self.axis}
        self.outputs = {'Indices': self.indices, 'Out': self.out}

    def init_axis(self):
        self.axis = -1

    def test_check_output(self):
        self.check_output()


class TestArgsortOpAxis0(TestArgsortOp):
    def init_axis(self):
        self.axis = 0


class TestArgsortOpAxis1(TestArgsortOp):
    def init_axis(self):
        self.axis = 1


class TestArgsortOpAxisNeg2(TestArgsortOp):
    def init_axis(self):
        self.axis = -2


if __name__ == "__main__":
    unittest.main()
