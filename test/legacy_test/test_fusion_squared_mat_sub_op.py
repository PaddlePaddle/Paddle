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


class TestFusionSquaredMatSubOp(OpTest):
    def setUp(self):
        self.op_type = 'fusion_squared_mat_sub'
        self.m = 11
        self.n = 12
        self.k = 4
        self.scalar = 0.5
        self.set_conf()
        matx = np.random.random((self.m, self.k)).astype("float32")
        maty = np.random.random((self.k, self.n)).astype("float32")

        self.inputs = {'X': matx, 'Y': maty}
        self.outputs = {
            'Out': (np.dot(matx, maty) ** 2 - np.dot(matx**2, maty**2))
            * self.scalar
        }
        self.attrs = {
            'scalar': self.scalar,
        }

    def set_conf(self):
        pass

    def test_check_output(self):
        self.check_output(check_dygraph=False)


class TestFusionSquaredMatSubOpCase1(TestFusionSquaredMatSubOp):
    def set_conf(self):
        self.scalar = -0.3


if __name__ == '__main__':
    unittest.main()
