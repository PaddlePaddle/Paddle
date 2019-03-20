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

import numpy as np
import unittest
from numpy import linalg as LA
from op_test import OpTest


class TestL2LossOp(OpTest):
    """Test squared_l2_norm
    """

    def setUp(self):
        self.op_type = "squared_l2_norm"
        self.max_relative_error = 0.05

        X = np.random.uniform(-1, 1, (13, 19)).astype("float32")
        X[np.abs(X) < self.max_relative_error] = 0.1
        self.inputs = {'X': X}
        self.outputs = {'Out': np.square(LA.norm(X))}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(
            ['X'], 'Out', max_relative_error=self.max_relative_error)


if __name__ == "__main__":
    unittest.main()
