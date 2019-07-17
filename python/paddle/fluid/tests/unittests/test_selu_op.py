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
import six
from op_test import OpTest


class SeluTest(OpTest):
    def setUp(self):
        self.op_type = "selu"
        self.x_shape = [3, 5, 5, 10]
        self.dtype = np.float64
        self.init_x_shape()
        self.init_dtype()

        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946

        x = np.random.normal(size=self.x_shape).astype(self.dtype)

        # Since zero point in selu is not differentiable, avoid randomize
        # zero.
        x[np.abs(x) < 0.005] = 0.02

        x_flat = x.flatten()

        for i in range(x_flat.size):
            if x_flat[i] < 0:
                x_flat[i] = alpha * np.exp(x_flat[i]) - alpha
            x_flat[i] = scale * x_flat[i]

        out_np = x_flat.reshape(self.x_shape)

        self.inputs = {'X': x}
        self.outputs = {'Out': out_np}

        self.attrs = {
            'alpha': alpha,
            'scale': scale,
        }

    def init_x_shape(self):
        pass

    def init_dtype(self):
        pass

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


if __name__ == "__main__":
    unittest.main()
