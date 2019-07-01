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
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.fluid.layers as layers
from op_test import OpTest
import gradient_checker
from decorator_helper import prog_scope


class TestInstagOp(OpTest):
    def setUp(self):
        self.op_type = 'instag'
        batch_size = 4
        x1_embed_size = 4
        fc_cnt = 2
        x1 = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1],
                       [1, 1, 1, 1]]).astype('double')

        x2 = np.array([[1, 2], [1, -1], [2, -1], [2, -1]]).astype('int64')

        x3 = np.array([[1, -1], [1, 2]]).astype('int64')

        out = np.array([
            [[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]],
            [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
        ]).astype('double')

        self.inputs = {
            'X1': x1,
            'X2': x2,
            'X3': x3,
        }

        self.outputs = {'Out': out, }

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X1'], 'Out', no_grad_set=set(['X2', 'X3']))

        def test_check_grad(self):
            self.check_grad(['X1'], 'Out', no_grad_set=set(['X2', 'X3']))


if __name__ == '__main__':
    unittest.main()
