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


class TestTopkOp(OpTest):
    def setUp(self):
        self.op_type = "top_k"
        k = 1
        input = np.random.random((32, 84)).astype("float32")
        output = np.ndarray((32, k))
        indices = np.ndarray((32, k)).astype("int64")

        self.inputs = {'X': input}
        self.attrs = {'k': k}

        for rowid in xrange(32):
            row = input[rowid]
            output[rowid] = np.sort(row)[-k:]
            indices[rowid] = row.argsort()[-k:]

        self.outputs = {'Out': output, 'Indices': indices}

    def test_check_output(self):
        self.check_output()


class TestTopkOp3d(OpTest):
    def setUp(self):
        self.op_type = "top_k"
        k = 1
        input = np.random.random((32, 2, 84)).astype("float32")
        input_flat_2d = input.reshape(64, 84)
        output = np.ndarray((64, k))
        indices = np.ndarray((64, k)).astype("int64")

        # FIXME: should use 'X': input for a 3d input
        self.inputs = {'X': input_flat_2d}
        self.attrs = {'k': k}

        for rowid in xrange(64):
            row = input_flat_2d[rowid]
            output[rowid] = np.sort(row)[-k:]
            indices[rowid] = row.argsort()[-k:]

        self.outputs = {'Out': output, 'Indices': indices}

    def test_check_output(self):
        self.check_output()


if __name__ == "__main__":
    unittest.main()
