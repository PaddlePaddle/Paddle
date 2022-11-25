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


def conv_shift_forward(x, y):
    out = np.zeros_like(x)
    M = x.shape[1]
    N = y.shape[1]
    y_half_width = (N - 1) // 2
    for i in range(M):
        for j in range(N):
            out[:, i] += x[:, (i + j + M - y_half_width) % M] * y[:, j]
    return out


class TestConvShiftOp(OpTest):

    def setUp(self):
        self.op_type = "conv_shift"

        batch_size = 10
        x_dim = 17
        y_dim = 11  # must be odd and <= x_dim
        x = np.random.random((batch_size, x_dim)).astype("float32")
        y = np.random.random((batch_size, y_dim)).astype("float32")
        self.inputs = {'X': x, 'Y': y}

        out = conv_shift_forward(x, y)
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['X', 'Y'], 'Out')

    def test_check_grad_ignore_x(self):
        self.check_grad(['Y'], 'Out')

    def test_check_grad_ignore_y(self):
        self.check_grad(['X'], 'Out')


if __name__ == '__main__':
    unittest.main()
