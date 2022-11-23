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


class TestProximalGDOp(OpTest):

    def setUp(self):
        self.op_type = "proximal_gd"
        w = np.random.random((102, 105)).astype("float32")
        g = np.random.random((102, 105)).astype("float32")
        lr = np.array([0.1]).astype("float32")
        l1 = 0.1
        l2 = 0.2

        self.inputs = {'Param': w, 'Grad': g, 'LearningRate': lr}
        self.attrs = {'l1': l1, 'l2': l2}
        prox_param = w - lr * g
        param_out = 0.0
        if l1 > 0.0:
            x = np.abs(prox_param) - lr * l1
            x[x < 0] = 0
            param_out = np.sign(prox_param) * (x / (1.0 + lr * l2))
        else:
            param_out = prox_param / (1.0 + lr * l2)

        self.outputs = {'ParamOut': param_out}

    def test_check_output(self):
        self.check_output()


if __name__ == "__main__":
    unittest.main()
