#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import unittest
import numpy as np
from op_test import OpTest


class TestElementwisePowOp(OpTest):
    def setUp(self):
        self.op_type = "elementwise_pow"
        self.inputs = {
            'X': np.random.uniform(0.1, 1, [13, 17]).astype("float32"),
            'Y': np.random.uniform(0.1, 1, [13, 17]).astype("float32")
        }
        self.outputs = {'Out': np.power(self.inputs['X'], self.inputs['Y'])}

    def test_check_output(self):
        self.check_output()


class TestElementwisePowOp_scalar(TestElementwisePowOp):
    def setUp(self):
        self.op_type = "elementwise_pow"
        self.inputs = {
            'X': np.random.rand(2, 3, 4).astype('float32'),
            'Y': np.random.rand(1).astype('float32')
        }
        self.outputs = {'Out': np.power(self.inputs['X'], self.inputs['Y'])}


if __name__ == '__main__':
    unittest.main()
