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
from scipy.special import erf
from op_test import OpTest, OpTestFp16


class TestErfOp(OpTest):
    def setUp(self):
        self.op_type = "erf"
        self.dtype = "float64"
        self.x_shape = [11, 17]
        x = np.random.uniform(-1, 1, size=self.x_shape).astype(self.dtype)
        y_ref = erf(x).astype(self.dtype)
        self.inputs = {'X': x}
        self.outputs = {'Out': y_ref}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


if __name__ == '__main__':
    unittest.main()
