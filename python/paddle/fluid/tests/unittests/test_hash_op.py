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


class TestScaleOp(OpTest):
    def setUp(self):
        self.op_type = "hash"
        self.init_test_case()
        self.inputs = {'X': (self.in_seq, self.lod)}
        self.attrs = {'num_hash': 8, 'mod_by': 10000}
        self.outputs = {'Out': (self.out_seq, self.lod)}

    def init_test_case(self):
        self.in_seq = np.random.randint(0, 10, (30, 1)).astype("int32")
        self.lod = [[9, 4, 11, 6]]
        self.out_seq = np.ones([30, 8], dtype=np.int32)

    def test_check_output(self):
        self.check_output()


if __name__ == "__main__":
    unittest.main()
