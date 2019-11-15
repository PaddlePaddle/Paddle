# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import paddle.fluid as fluid
from paddle.fluid.framework import Variable
from paddle.fluid.core import VarBase
import paddle.fluid.dygraph as dygraph
import numpy as np
import unittest


class TestIsInstanceVariable(unittest.TestCase):
    def test_main(self):
        with dygraph.guard():
            var_base = dygraph.to_variable(np.array([3, 4, 5]))._ivar
            self.assertTrue(isinstance(var_base, VarBase))
            self.assertTrue(isinstance(var_base, Variable))
            self.assertEqual(type(var_base), VarBase)

        var = fluid.data(name='x', shape=[None, 1], dtype='float32')
        self.assertFalse(isinstance(var, VarBase))
        self.assertTrue(isinstance(var, Variable))
        self.assertEqual(type(var), Variable)


if __name__ == '__main__':
    unittest.main()
