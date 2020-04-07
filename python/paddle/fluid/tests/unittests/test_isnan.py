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

import paddle.tensor as tensor
import paddle.fluid.dygraph as dg
import paddle.fluid as fluid


class TestIsnanOp(unittest.TestCase):
    def test_isnan(self):
        self.op_type = "isnan"
        x = np.array([2, 3, 3, 1, 5, 3], dtype='float32')
        y_ref = [False]
        place = fluid.CPUPlace()
        with dg.guard(place) as g:
            x_var = dg.to_variable(x)
            y_var = tensor.isnan(x_var)
            y_test = y_var.numpy().tolist()
        self.assertTrue(y_test == y_ref)


if __name__ == '__main__':
    unittest.main()
