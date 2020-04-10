#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from paddle import tensor
import paddle.fluid as fluid
from paddle.fluid import compiler, Program, program_guard


class TestRandOpError(unittest.TestCase):
    def test_errors(self):
        main_prog = Program()
        start_prog = Program()
        with program_guard(main_prog, start_prog):

            def test_Variable():
                x1 = fluid.create_lod_tensor(
                    np.zeros((4, 784)), [[1, 1, 1, 1]], fluid.CPUPlace())
                tensor.rand(x1)

            self.assertRaises(TypeError, test_Variable)

            def test_dtype():
                dim_1 = fluid.layers.fill_constant([1], "int64", 3)
                dim_2 = fluid.layers.fill_constant([1], "int32", 5)
                tensor.rand(shape=[dim_1, dim_2], dtype='int32')

            self.assertRaises(TypeError, test_dtype)


if __name__ == "__main__":
    unittest.main()
