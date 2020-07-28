# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
from paddle import fluid
import paddle.fluid.dygraph as dg
import unittest


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def glu(x, dim=-1):
    a, b = np.split(x, 2, axis=dim)
    out = a * sigmoid(b)
    return out


class TestGLUCase(unittest.TestCase):
    def setUp(self):
        self.x = np.random.randn(5, 20)
        self.dim = -1
        self.out = glu(self.x, self.dim)

    def check_identity(self, place):
        with dg.guard(place):
            x_var = dg.to_variable(self.x)
            y_var = fluid.nets.glu(x_var, self.dim)
            y_np = y_var.numpy()

        np.testing.assert_allclose(y_np, self.out)

    def test_case(self):
        self.check_identity(fluid.CPUPlace())
        if fluid.is_compiled_with_cuda():
            self.check_identity(fluid.CUDAPlace(0))


if __name__ == '__main__':
    unittest.main()
