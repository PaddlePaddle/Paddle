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
from op_test import OpTest

import paddle.fluid.core as core
from paddle.fluid.op import Operator
import paddle.fluid as fluid
from paddle.fluid import compiler, Program, program_guard


class TestZerosOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):
            # The input dtype of zeros_op must be bool, float16, float32, float64, int32, int64.
            shape = [4]
            dtype = "int8"
            self.assertRaises(TypeError, fluid.layers.zeros, shape, dtype)


if __name__ == "__main__":
    unittest.main()
