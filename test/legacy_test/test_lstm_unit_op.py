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


def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))


def tanh_np(x):
    return 2 * sigmoid_np(2.0 * x) - 1.0


class LstmUnitTest(OpTest):
    def setUp(self):
        self.op_type = "lstm_unit"
        x_np = np.random.normal(size=(15, 160)).astype("float64")
        c_np = np.random.normal(size=(15, 40)).astype("float64")
        i_np, f_np, o_np, j_np = np.split(x_np, 4, axis=1)
        forget_bias_np = 0.0
        self.attrs = {'forget_bias': 0.0}

        new_c = c_np * sigmoid_np(f_np + forget_bias_np) + sigmoid_np(
            i_np
        ) * tanh_np(j_np)
        new_h = tanh_np(new_c) * sigmoid_np(o_np)

        self.inputs = {'X': x_np, 'C_prev': c_np}
        self.outputs = {'C': new_c, 'H': new_h}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X', 'C_prev'], ['C', 'H'])


if __name__ == "__main__":
    unittest.main()
