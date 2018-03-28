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


class TestSplitIdsOp(OpTest):
    def setUp(self):
        self.op_type = "split_ids"
        ids = np.array([[0], [2], [2], [3], [5], [5], [6]]).astype('int64')
        out0 = np.array([[0], [3], [6]]).astype('int64')
        out1 = np.array([[]]).astype('int64')
        out2 = np.array([[2], [2], [5], [5]]).astype('int64')
        self.inputs = {'Ids': ids}
        self.outputs = {'Out': [('out0', out0), ('out1', out1), ('out2', out2)]}

    def test_check_output(self):
        self.check_output()


if __name__ == '__main__':
    unittest.main()
