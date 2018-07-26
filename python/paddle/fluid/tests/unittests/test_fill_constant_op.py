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


class TestFillConstantOp1(OpTest):
    def setUp(self):
        '''Test fill_constant op with specified value
        '''
        self.op_type = "fill_constant"

        self.inputs = {}
        self.attrs = {'shape': [123, 92], 'value': 3.8}
        self.outputs = {'Out': np.full((123, 92), 3.8)}

    def test_check_output(self):
        self.check_output()


class TestFillConstantOp2(OpTest):
    def setUp(self):
        '''Test fill_constant op with default value
        '''
        self.op_type = "fill_constant"

        self.inputs = {}
        self.attrs = {'shape': [123, 92]}
        self.outputs = {'Out': np.full((123, 92), 0.0)}

    def test_check_output(self):
        self.check_output()


if __name__ == "__main__":
    unittest.main()
