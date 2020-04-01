# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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


class TestDiagEmbedOp(OpTest):
    def setUp(self):
        self.op_type = "diag_embed"
        self.init_config()
        self.outputs = {'Out': self.target}

    def test_check_output(self):
        self.check_output()

    def init_config(self):
        self.case = np.random.randn(2, 3).astype('float32')
        self.inputs = {'Input': self.case}
        self.attrs = {'offset': 0, 'dim1': -2, 'dim2': -1}
        self.target = np.stack([np.diag(r, 0) for r in self.inputs['Input']], 0)


class TestDiagEmbedOpCase1(TestDiagEmbedOp):
    def init_config(self):
        self.case = np.random.randn(2, 3).astype('float32')
        self.inputs = {'Input': self.case}
        self.attrs = {'offset': -1, 'dim1': 0, 'dim2': 2}
        self.target = np.stack([np.diag(r, -1) for r in self.inputs['Input']],
                               1)


if __name__ == "__main__":
    unittest.main()
