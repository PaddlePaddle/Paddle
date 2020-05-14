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
import paddle.fluid as fluid
import numpy as np


class TestImperativeCheckNanInf(unittest.TestCase):
    def setUp(self):
        self.a_np = np.array([1]).astype(np.float32)
        self.b_np = np.array([0]).astype(np.float32)
        self.c_np = np.array([0]).astype(np.float32)
        fluid.set_flags({"FLAGS_imperative_check_nan_inf": 1})

    def test_normal_run(self):
        with fluid.dygraph.guard():
            a = fluid.dygraph.to_variable(self.a_np)
            b = fluid.dygraph.to_variable(self.b_np)
            out = fluid.layers.elementwise_div(b, a)

    def test_inf(self):
        with fluid.dygraph.guard():
            a = fluid.dygraph.to_variable(self.a_np)
            b = fluid.dygraph.to_variable(self.b_np)
            self.assertRaises(Exception, fluid.layers.elementwise_div, a, b)

    def test_nan(self):
        with fluid.dygraph.guard():
            b = fluid.dygraph.to_variable(self.b_np)
            c = fluid.dygraph.to_variable(self.c_np)
            self.assertRaises(Exception, fluid.layers.elementwise_div, b, c)


if __name__ == '__main__':
    unittest.main()
