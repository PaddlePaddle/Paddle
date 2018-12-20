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


class TestAccuracyOp(OpTest):
    def setUp(self):
        self.op_type = "accuracy"
        n = 8192
        infer = np.random.random((n, 1)).astype("float32")
        indices = np.random.randint(0, 2, (n, 1))
        label = np.random.randint(0, 2, (n, 1))
        self.inputs = {'Out': infer, 'Indices': indices, "Label": label}
        num_correct = 0
        for rowid in range(n):
            for ele in indices[rowid]:
                if ele == label[rowid]:
                    num_correct += 1
                    break
        self.outputs = {
            'Accuracy': np.array([num_correct / float(n)]).astype("float32"),
            'Correct': np.array([num_correct]).astype("int32"),
            'Total': np.array([n]).astype("int32")
        }

    def test_check_output(self):
        self.check_output()


if __name__ == '__main__':
    unittest.main()
