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

from __future__ import print_function

import unittest, sys
sys.path.append("../")
import numpy as np
from op_test import OpTest


class TestNGRAPHIncrementOp(OpTest):
    def setUp(self):
        self.op_type = "increment"
        self.dtype = np.float32
        self.init_dtype_type()
        self.inputs = {'X': np.random.random(1).astype(self.dtype)}
        self.attrs = {'step': 2.0}
        self.outputs = {
            'Out': self.inputs['X'] + self.dtype(self.attrs['step'])
        }
        self._cpu_only = True

    def init_dtype_type(self):
        pass

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', check_dygraph=False)


if __name__ == "__main__":
    unittest.main()
