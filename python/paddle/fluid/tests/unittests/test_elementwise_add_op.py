#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import unittest
import numpy as np
import paddle
import paddle.fluid.core as core
from paddle.fluid.tests.unittests.op_test import OpTest, skip_check_grad_ci, convert_float_to_uint16
import paddle.fluid as fluid
from paddle.fluid import compiler, Program, program_guard


class TestBoolAddFloatElementwiseAddop(unittest.TestCase):

    def test_dygraph_add(self):
        paddle.disable_static()
        # a = 1.5
        # b = paddle.full([4, 5, 6], True, dtype='bool')
        # c = a + b
        # self.assertTrue(c.dtype == core.VarDesc.VarType.FP32)

        np_a = np.random.random((2, 3, 4)).astype(np.float64)
        np_b = np.random.random((2, 3, 4)).astype(np.float64)
        expect_out = np_a + np_b

        tensor_a = paddle.to_tensor(np_a, dtype="float64")
        tensor_b = paddle.to_tensor(np_b, dtype="float64")
        actual_out = 1 + tensor_b
        print(tensor_a)
        print(actual_out)
        # np.testing.assert_allclose(actual_out, expect_out)


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
