# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
from eager_op_test import OpTest


class TestCopysignOp(OpTest):
    def setUp(self):
        self.op_type = "copysign"
        self.initTestCase()
        self.inputs = {
            'X': np.random.random(self.shape).astype(self.dtype),
            'Y': np.random.random(self.shape).astype(self.dtype),
        }
        self.init_output()

    def initTestCase(self):
        self.dtype = np.float64
        self.shape = (100, 100)

    def init_output(self):
        x = self.inputs['X']
        y = self.inputs['Y']
        self.outputs = np.copysign(x, y)

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['X', 'Y'], 'Out')


class TestCopysignOpTest(TestCopysignOp):
    def initTestCase(self):
        self.dtype = np.float64
        self.shape = np.shape
