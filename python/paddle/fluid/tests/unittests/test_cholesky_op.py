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


class TestCholeskyOp(OpTest):
    def setUp(self):
        self.op_type = "cholesky"
        self._input_shape = (8, 32, 32)
        self._upper = True
        self.init_config()
        trans_dims = list(range(len(self._input_shape) - 2)) + [
            len(self._input_shape) - 1, len(self._input_shape) - 2
        ]
        l = np.random.random(self._input_shape).astype("float64")
        # construct symmetric positive-definite matrice
        input_data = np.matmul(l, l.transpose(trans_dims)) + 1e-03
        output_data = np.linalg.cholesky(input_data).astype("float64")
        if self._upper:
            output_data = output_data.transpose(trans_dims)
        self.inputs = {"X": input_data}
        self.attrs = {"upper": self._upper}
        self.outputs = {"Out": output_data}

    def test_check_output(self):
        self.check_output()

    def init_config(self):
        self._upper = True


class TestCholeskyOpLower(TestCholeskyOp):
    def init_config(self):
        self._upper = False


class TestCholeskyOp2D(TestCholeskyOp):
    def init_config(self):
        self._input_shape = (64, 64)


if __name__ == "__main__":
    unittest.main()
