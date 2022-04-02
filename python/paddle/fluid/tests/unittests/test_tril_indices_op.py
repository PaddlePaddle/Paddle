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

from __future__ import print_function

import unittest
import numpy as np
from op_test import OpTest
import paddle.nn.functional as F
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.tensor as tensor
import paddle


class TestTrilIndicesOp(OpTest):
    def setUp(self):
        self.op_type = "tril_indices"
        self.inputs = {}
        self.init_config()
        self.outputs = {'out': self.target}

    def test_check_output(self):
        self.check_output()

    def init_config(self):
        self.attrs = {'rows': 0, 'cols': 0, 'offset': 0}
        self.target = np.tril_indices(self.attrs['rows'],self.attrs['offset'],self.attrs['cols'])


class TestTrilIndicesOpCase1(TestTrilIndicesOp):
    def init_config(self):
        self.attrs = {'rows': 4, 'cols': 4, 'offset': 0}
        self.target = np.tril_indices(self.attrs['rows'],self.attrs['offset'],self.attrs['cols'])


class TestTrilIndicesOpCase2(TestTrilIndicesOp):
    def init_config(self):
        self.attrs = {'rows': 4, 'cols': 4, 'offset': 2}
        self.target = np.tril_indices(self.attrs['rows'],self.attrs['offset'],self.attrs['cols'])


class TestTrilIndicesAPICase(unittest.TestCase):
    def test_case1(self):
        
        out1 = paddle.tril_indices(4,4,0)
        out2 = paddle.tril_indices(4,4,2)

        place = core.CPUPlace()
        exe = fluid.Executor(place)
        results = exe.run(fluid.default_main_program(),
                          fetch_list=[out1, out2],
                          return_numpy=True)
        target1 = np.tril_indices(4,0,4)
        target2 = np.tril_indices(4,2,4)
        self.assertTrue(np.allclose(results[0], target1))
        self.assertTrue(np.allclose(results[1], target2))

if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
