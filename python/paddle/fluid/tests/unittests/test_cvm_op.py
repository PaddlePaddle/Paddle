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

import numpy as np
from math import log
from math import exp
from op_test import OpTest
import unittest


class TestCVMOp(OpTest):
    """
        Test cvm op with discrete one-hot labels.
    """

    def setUp(self):
        self.op_type = "cvm"
        batch_size = 4
        dims = 11
        lod = [[1]]
        self.inputs = {
            'X': (np.random.uniform(0, 1, [1, dims]).astype("float32"), lod),
            'CVM': np.array([[0.6, 0.4]]).astype("float32"),
        }
        self.attrs = {'use_cvm': False}
        out = []
        for index, emb in enumerate(self.inputs["X"][0]):
            out.append(emb[2:])
        self.outputs = {'Y': (np.array(out), lod)}

    def test_check_output(self):
        self.check_output()


if __name__ == '__main__':
    unittest.main()
