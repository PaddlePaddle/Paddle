#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from op_test import OpTest


class TestFlipOp(OpTest):
    def setUp(self):
        self.op_type = 'flip'
        self.init_test_case()
        self.inputs = {'X': np.random.random(self.in_shape).astype('float64')}
        self.init_attrs()
        self.outputs = {'Out': self.calc_ref_res()}

    def init_attrs(self):
        self.attrs = {"dims": self.dims}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X"], "Out")

    def init_test_case(self):
        self.in_shape = (6, 4, 2, 3)
        self.dims = [0, 1]

    def calc_ref_res(self):
        res = self.inputs['X']
        for axis in self.dims:
            res = np.flip(res, axis)
        return res


class TestFlipOpAxis1(TestFlipOp):
    def init_test_case(self):
        self.in_shape = (2, 4, 4)
        self.dims = [0]


class TestFlipOpAxis2(TestFlipOp):
    def init_test_case(self):
        self.in_shape = (4, 4, 6, 3)
        self.dims = [0, 2]


class TestFlipOpAxis3(TestFlipOp):
    def init_test_case(self):
        self.in_shape = (4, 3, 1)
        self.dims = [0, 1, 2]


class TestFlipOpAxis4(TestFlipOp):
    def init_test_case(self):
        self.in_shape = (6, 4, 2, 2)
        self.dims = [0, 1, 2, 3]


if __name__ == "__main__":
    unittest.main()
