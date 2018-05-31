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


class PReluTest(OpTest):
    def setUp(self):
        self.op_type = "prelu"
        x_np = np.random.normal(size=(10, 10)).astype("float32")

        for pos, val in np.ndenumerate(x_np):
            # Since zero point in prelu is not differentiable, avoid randomize
            # zero.
            while abs(val) < 1e-3:
                x_np[pos] = np.random.normal()
                val = x_np[pos]

        x_np_sign = np.sign(x_np)
        x_np = x_np_sign * np.maximum(x_np, .005)
        alpha_np = np.array([.1], dtype="float32")
        self.inputs = {'X': x_np, 'Alpha': alpha_np}
        out_np = np.maximum(self.inputs['X'], 0.)
        out_np = out_np + np.minimum(self.inputs['X'],
                                     0.) * self.inputs['Alpha']
        assert out_np is not self.inputs['X']
        self.outputs = {'Out': out_np}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


if __name__ == "__main__":
    unittest.main()
