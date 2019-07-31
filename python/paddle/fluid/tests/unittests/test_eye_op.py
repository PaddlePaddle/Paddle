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
import numpy as np
from op_test import OpTest

import paddle.fluid.framework as framework


class TestEyeOp(OpTest):
    def setUp(self):
        '''
	Test eye op with specified shape
        '''
        self.op_type = "eye"

        self.inputs = {}
        self.attrs = {
            'num_rows': 2019,
            'num_columns': 3019,
            'dtype': framework.convert_np_dtype_to_dtype_(np.int32)
        }
        self.outputs = {'Out': np.eye(2019, 3019, dtype=np.int32)}

    def test_check_output(self):
        self.check_output()


class TestEyeOp1(OpTest):
    def setUp(self):
        '''
	Test eye op with default parameters
        '''
        self.op_type = "eye"

        self.inputs = {}
        self.attrs = {'num_rows': 2019}
        self.outputs = {'Out': np.eye(2019, dtype=float)}

    def test_check_output(self):
        self.check_output()


class TestEyeOp2(OpTest):
    def setUp(self):
        '''
        Test eye op with specified shape
        '''
        self.op_type = "eye"

        self.inputs = {}
        self.attrs = {'num_rows': 2019, 'num_columns': 1}
        self.outputs = {'Out': np.eye(2019, 1, dtype=float)}

    def test_check_output(self):
        self.check_output()


if __name__ == "__main__":
    unittest.main()
