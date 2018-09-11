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


class TestBilinearTensorProductOp(OpTest):
    def setUp(self):
        self.op_type = "bilinear_tensor_product"
        batch_size = 6
        size0 = 3
        size1 = 4
        size2 = 5
        a = np.random.random((batch_size, size0)).astype("float32")
        b = np.random.random((batch_size, size1)).astype("float32")
        w = np.random.random((size2, size0, size1)).astype("float32")
        bias = np.random.random((1, size2)).astype("float32")
        output = np.zeros((batch_size, size2)).astype("float32")
        for i in range(size2):
            w_i = w[i, :, :]
            output[:, i] = np.sum(np.matmul(a, w_i) * b, axis=1)
        self.inputs = {
            'X': a,
            'Y': b,
            'Weight': w,
            'Bias': bias,
        }
        self.outputs = {'Out': output + bias}

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['X', 'Y', 'Weight', 'Bias'], 'Out')


if __name__ == "__main__":
    unittest.main()
