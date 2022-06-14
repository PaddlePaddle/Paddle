#  Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import numpy as np
import unittest
import sys
from paddle.fluid.tests.unittests.op_test import OpTest
import paddle
import paddle.fluid as fluid

paddle.enable_static()
SEED = 2021


class TestMLUSigmoid(OpTest):
    def setUp(self):
        self.op_type = "sigmoid"
        self.set_mlu()
        self.init_dtype()

        np.random.seed(SEED)
        x = np.random.uniform(-1, 1, [11, 17]).astype(self.dtype)
        out = 1 / (1 + np.exp(-x))

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        self.check_grad_with_place(
            self.place, ['X'], 'Out', max_relative_error=0.01)

    def set_mlu(self):
        self.__class__.use_mlu = True
        self.place = paddle.MLUPlace(0)

    def init_dtype(self):
        self.dtype = np.float32


class TestMLUSigmoidFp16(TestMLUSigmoid):
    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-3)

    def init_dtype(self):
        self.dtype = np.float16


if __name__ == '__main__':
    unittest.main()
