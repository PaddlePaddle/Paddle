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

from paddle.fluid.op import Operator
import paddle.fluid.core as core
import unittest
import numpy as np


class HasDataOpTester(unittest.TestCase):
    def setUp(self):
        self.scope = core.Scope()
        self.scope.var('X')
        self.scope.var('Out')
        self.place = core.CPUPlace()
        x_data = np.array([1])
        x_tensor = self.scope.var('X').get_tensor()
        x_tensor.set(x_data, self.place)
        out_tensor = self.scope.var('Out').get_tensor()

    def test_run(self):
        op = Operator('has_data', X='X', Out='Out')
        op.run(self.scope, self.place)
        out_tensor = self.scope.find_var('Out').get_tensor()
        print 'output: ', np.array(out_tensor)


class HasDataOpGPUTester(HasDataOpTester):
    def setUp(self):
        self.scope = core.Scope()
        self.scope.var('X')
        self.scope.var('Out')
        self.place = core.CUDAPlace(0)
        x_data = np.array([])
        x_tensor = self.scope.var('X').get_tensor()
        x_tensor.set(x_data, self.place)
        out_tensor = self.scope.var('Out').get_tensor()


if __name__ == '__main__':
    unittest.main()
