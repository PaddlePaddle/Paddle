#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import sys

sys.path.append('..')
from op_test import OpTest
import paddle.fluid.core as core
import paddle.fluid as fluid
from paddle.fluid import compiler, Program, program_guard
import paddle
import paddle.nn.functional as F

paddle.enable_static()
np.random.seed(10)


class TestSqrt(OpTest):

    def setUp(self):
        self.op_type = "sqrt"
        self.dtype = 'float32'
        self.set_mlu()
        self.python_api = paddle.sqrt

        np.random.seed(1023)
        x = np.random.uniform(0.1, 1, [11, 17]).astype(self.dtype)
        out = np.sqrt(x)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def set_mlu(self):
        self.__class__.use_mlu = True
        self.place = paddle.device.MLUPlace(0)

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ['X'], 'Out', check_eager=False)

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestSqrtHalf(OpTest):

    def setUp(self):
        self.op_type = "sqrt"
        self.dtype = 'float16'
        self.set_mlu()
        self.python_api = paddle.sqrt

        np.random.seed(1023)
        x = np.random.uniform(0.1, 1, [11, 17]).astype(self.dtype)
        out = np.sqrt(x)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def set_mlu(self):
        self.__class__.use_mlu = True
        self.place = paddle.device.MLUPlace(0)

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ['X'],
                                   'Out',
                                   check_eager=False,
                                   max_relative_error=0.85)

    def test_check_output(self):
        self.check_output_with_place(self.place)


if __name__ == "__main__":
    unittest.main()
