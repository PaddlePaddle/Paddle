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

import paddle
import unittest
import numpy as np
from op_test import OpTest
from functools import partial


def arange_wrapper(start, end, step, dtype=None):
    return paddle.arange(start, end, step, dtype)


class TestRangeOp(OpTest):
    def setUp(self):
        self.op_type = "range"
        self.init_config()
        self.inputs = {
            'Start': np.array([self.case[0]]).astype(self.dtype),
            'End': np.array([self.case[1]]).astype(self.dtype),
            'Step': np.array([self.case[2]]).astype(self.dtype)
        }

        self.outputs = {
            'Out': np.arange(self.case[0], self.case[1],
                             self.case[2]).astype(self.dtype)
        }

    def init_config(self):
        self.dtype = np.float32
        self.python_api = partial(arange_wrapper, dtype=self.dtype)
        self.case = (0, 1, 0.2)

    def test_check_output(self):
        self.check_output(check_eager=True)


class TestFloatRangeOpCase0(TestRangeOp):
    def init_config(self):
        self.dtype = np.float32
        self.python_api = partial(arange_wrapper, dtype=self.dtype)
        self.case = (0, 5, 1)


class TestInt32RangeOpCase0(TestRangeOp):
    def init_config(self):
        self.dtype = np.int32
        self.python_api = partial(arange_wrapper, dtype=self.dtype)
        self.case = (0, 5, 2)


class TestInt32RangeOpCase1(TestRangeOp):
    def init_config(self):
        self.dtype = np.int32
        self.python_api = partial(arange_wrapper, dtype=self.dtype)
        self.case = (10, 1, -2)


class TestInt32RangeOpCase2(TestRangeOp):
    def init_config(self):
        self.dtype = np.int32
        self.python_api = partial(arange_wrapper, dtype=self.dtype)
        self.case = (-1, -10, -2)


if __name__ == "__main__":
    unittest.main()
