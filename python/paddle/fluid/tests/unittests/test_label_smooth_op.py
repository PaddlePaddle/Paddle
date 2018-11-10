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


class TestLabelSmoothOp(OpTest):
    def config(self):
        self.op_type = "label_smooth"
        self.epsilon = 0.1
        batch_size, self.label_dim = 5, 10
        self.label = np.zeros((batch_size, self.label_dim)).astype("float64")
        nonzero_index = np.random.randint(self.label_dim, size=(batch_size))
        self.label[np.arange(batch_size), nonzero_index] = 1

    def setUp(self):
        self.config()
        smoothed_label = (1 - self.epsilon
                          ) * self.label + self.epsilon / self.label_dim
        self.inputs = {'X': self.label}
        self.attrs = {'epsilon': self.epsilon}
        self.outputs = {'Out': smoothed_label}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X"], "Out")


class TestLabelSmoothOpWithPriorDist(TestLabelSmoothOp):
    def setUp(self):
        self.config()
        dist = np.random.random((1, self.label_dim))
        smoothed_label = (1 - self.epsilon) * self.label + self.epsilon * dist
        self.inputs = {'X': self.label, 'PriorDist': dist}
        self.attrs = {'epsilon': self.epsilon}
        self.outputs = {'Out': smoothed_label}


if __name__ == '__main__':
    unittest.main()
