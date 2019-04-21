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


class TestGatherOp(OpTest):
    def setUp(self):
        self.test_gc = True
        self.op_type = "gather"
        self.config()
        xnp = np.random.random(self.x_shape).astype("float32")
        self.inputs = {'X': xnp, 'Index': np.array(self.index).astype("int32")}
        self.outputs = {'Out': self.inputs["X"][self.inputs["Index"]]}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')

    def config(self):
        self.x_shape = (10, 20)
        self.index = [1, 3, 5]


class TestCase1(TestGatherOp):
    def config(self):
        self.x_shape = (10)
        self.index = [1, 3, 5]


if __name__ == "__main__":
    unittest.main()
