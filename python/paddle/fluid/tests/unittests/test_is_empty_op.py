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


class TestEmpty(OpTest):
    def setUp(self):
        self.op_type = "is_empty"
        self.inputs = {'X': np.array([1, 2, 3])}
        self.outputs = {'Out': np.array([False])}

    def test_check_output(self):
        self.check_output()


class TestNotEmpty(TestEmpty):
    def setUp(self):
        self.op_type = "is_empty"
        self.inputs = {'X': np.array([])}
        self.outputs = {'Out': np.array([True])}


if __name__ == "__main__":
    unittest.main()
