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

from __future__ import print_function

import unittest
import numpy as np
from op_test import OpTest


class TestDiagOp(OpTest):
    def setUp(self):
        self.op_type = "diag"
        self.init_config()
        self.inputs = {'Diagonal': self.case}

        self.outputs = {'Out': np.diag(self.inputs['Diagonal'])}

    def test_check_output(self):
        self.check_output()

    def init_config(self):
        self.case = np.arange(3, 6)


class TestDiagOpCase1(TestDiagOp):
    def init_config(self):
        self.case = np.array([3], dtype='int32')


if __name__ == "__main__":
    unittest.main()
