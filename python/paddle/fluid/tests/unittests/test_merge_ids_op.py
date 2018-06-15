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


class TestMergeIdsOp(OpTest):
    def setUp(self):
        self.op_type = "merge_ids"
        ids = np.array([[0], [2], [2], [3], [5], [5], [6]]).astype('int64')
        x0 = np.array([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]]).astype('float32')
        x1 = np.array([]).astype('float32')
        x2 = np.array([[0.4, 0.5], [0.4, 0.5], [0.5, 0.6],
                       [0.5, 0.6]]).astype('float32')
        out = np.array([[0.1, 0.2], [0.4, 0.5], [0.4, 0.5], [0.2, 0.3],
                        [0.5, 0.6], [0.5, 0.6], [0.3, 0.4]]).astype('float32')
        self.inputs = {'Ids': ids, "X": [('x0', x0), ('x1', x1), ('x2', x2)]}
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output()


if __name__ == '__main__':
    unittest.main()
