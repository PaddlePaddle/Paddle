#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import paddle
import paddle.fluid as fluid


def gather_numpy(x, axis, index):
    result = x[:, index, :]
    return result


class TestGatherOp(OpTest):
    def setUp(self):
        self.op_type = "gather_v2"
        self.config()
        xnp = np.random.random(self.x_shape).astype(self.x_type)
        axis_np = np.array(self.axis).astype(self.index_type)
        index_np = np.array(self.index).astype(self.index_type)
        self.inputs = {'X': xnp, 'Index': index_np, 'Axis': axis_np}
        out = xnp[:, self.index, :]
        print(out)
        self.outputs = {'Y': out}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Y')

    def config(self):
        """
        For multi-dimension input
        """
        self.x_shape = (3, 88, 3)
        self.x_type = "float64"
        self.index = [1, 1, 1]
        self.index_type = "int32"
        self.axis = [1]
        self.axis_type = "int32"


if __name__ == "__main__":
    unittest.main()
