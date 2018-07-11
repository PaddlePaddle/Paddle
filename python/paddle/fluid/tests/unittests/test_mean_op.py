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


class TestMeanOp(OpTest):
    def setUp(self):
        self.op_type = "mean"
        self.dtype = np.float32
        self.init_dtype()

        self.inputs = {'X': np.random.random((2, 3)).astype(self.dtype)}
        self.outputs = {'Out': np.mean(self.inputs["X"])}

    def init_dtype(self):
        pass

    def test_check_output(self):
        self.check_output()

    def test_checkout_grad(self):
        self.check_grad(['X'], 'Out')


class TestMeanFP16Op(TestMeanOp):
    def init_dtype(self):
        self.dtype = np.float16

    def test_check_output(self):
        self.check_output(atol=0.05)


if __name__ == "__main__":
    unittest.main()
