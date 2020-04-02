# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid as fluid


def dist(x, y, p):
    if p == 0.:
        out = np.count_nonzero(x - y)
    elif p == float("inf"):
        out = np.max(np.abs(x - y))
    elif p == float("-inf"):
        out = np.min(np.abs(x - y))
    else:
        out = np.power(np.sum(np.power(np.abs(x - y), p)), 1.0 / p)
    return np.array(out).astype(x.dtype)


class TestDistOp(OpTest):
    def setUp(self):
        self.op_type = 'dist'
        self.attrs = {}
        self.init_case()
        self.inputs = {
            "X": np.random.random(self.x_shape).astype("float64"),
            "Y": np.random.random(self.y_shape).astype("float64")
        }
        """
        self.inputs = {
            "X": np.array([[3, 3], [1, 3]]).astype("float64"),
            "Y": np.array([[1, 1], [3, 1]]).astype("float64")
        }
        """
        self.attrs["p"] = self.p
        self.outputs = {
            "Out": dist(self.inputs["X"], self.inputs["Y"], self.attrs["p"])
        }

    def init_case(self):
        self.x_shape = (2, 5, 5, 6)
        self.y_shape = (5, 5, 6)
        self.p = 3

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X", "Y"], "Out")


"""
class TestDistOpNorm1(TestDistOp):
    def init_case(self):
        self.x_shape = (5, 5, 6)
        self.y_shape = (5, 5, 6)
        self.p = 1.

class TestDistOpNorm2(TestDistOp):
    def init_case(self):
        self.x_shape = (5, 5, 6)
        self.y_shape = (5, 5, 6)
        self.p = 2.


class TestDistOpNorm3(TestDistOp):
    def init_case(self):
        self.x_shape = (5, 5, 6)
        self.y_shape = (5, 5, 6)
        self.p = 3.

class TestDistOpNormFloat(TestDistOp):
    def init_case(self):
        self.x_shape = (5, 5, 6)
        self.y_shape = (5, 5, 6)
        self.p = 5.6
"""

if __name__ == '__main__':
    unittest.main()
