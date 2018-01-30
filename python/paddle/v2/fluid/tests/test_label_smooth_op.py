#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
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


class TestLabelSmoothOp(OpTest):
    def setUp(self):
        self.op_type = "label_smooth"
        epsilon = 0.1
        batch_size, label_dim = 5, 10
        label = np.zeros((batch_size, label_dim)).astype("float64")
        nonzero_index = np.random.randint(label_dim, size=(batch_size))
        label[np.arange(batch_size), nonzero_index] = 1
        smoothed_label = (1 - epsilon) * label + epsilon / label_dim
        self.inputs = {'X': label}
        self.attrs = {'epsilon': epsilon}
        self.outputs = {'Out': smoothed_label}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X"], "Out")


if __name__ == '__main__':
    unittest.main()
