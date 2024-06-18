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

from paddle import _C_ops


def l1_norm_wrapper(x):
    return _C_ops.l1_norm(x)


class TestL1NormOp(OpTest):
    """Test l1_norm"""

    def setUp(self):
        self.op_type = "l1_norm"
        self.python_api = l1_norm_wrapper
        self.max_relative_error = 0.005

        X = np.random.uniform(-1, 1, (13, 19)).astype("float32")
        X[np.abs(X) < self.max_relative_error] = 0.1
        self.inputs = {'X': X}
        self.outputs = {'Out': np.sum(np.abs(X))}

    def test_check_output(self):
        self.check_output(atol=2e-5, rtol=2e-5, inplace_atol=2e-5)

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


if __name__ == "__main__":
    unittest.main()
