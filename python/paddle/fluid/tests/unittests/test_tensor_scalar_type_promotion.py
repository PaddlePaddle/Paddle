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

import paddle

# deprecated module import
from paddle.fluid import core

# Related dtypes:
#  - int type: uint8, int8, int16, int32, int64
#  - float type: float32, float64
# Python scalar: 
#  - int
#  - float
# Numpy scalar:
#  - many


class TestTensorScalarTypePromotion(unittest.TestCase):
    def setUp(self):
        paddle.set_device('cpu')

    def test_int32_no_promote(self):
        a = a = paddle.ones([3, 3, 3], dtype='int32')
        b = 1  # int64 in python3
        c_add = a + b
        print(c_add)
        self.assertTrue(np.array_equal(c_add, a + a))
        # self.assertEqual(c_add.dtype, core.VarDesc.VarType.INT32)

    # def test_float32_no_promote(self):
    #     a = paddle.ones([3, 3, 3], dtype='float32')
    #     b = 2.0 # double in python3
    #     c_add = a + b
    #     print(c_add)
    #     self.assertEqual(c_add.dtype, core.VarDesc.VarType.FP32)

    # def test_int32_promote_to_int64(self):
    #     a = paddle.ones([3, 3, 3], dtype='int32')
    #     b = 1

    #     c_add = a + b
    #     self.assertTrue(np.array_equal(c_add, a + a))
    #     self.assertEqual(c_add.dtype, core.VarDesc.VarType.INT32) # INT64?

    #     c_sub = a - b
    #     self.assertTrue(np.array_equal(c_sub, a - a))
    #     self.assertEqual(c_add.dtype, core.VarDesc.VarType.INT32) # INT64?

    # def test_int32_promote_to_float32(self):
    #     a = paddle.ones([3, 3, 3], dtype='int32')

    #     # AssertionError: VarType.INT32 != VarType.FP32
    #     b = 1.0
    #     c_add = a + b
    #     print(c_add)
    #     self.assertEqual(c_add.dtype, core.VarDesc.VarType.FP32)

    # AssertionError: float value 1.5 cannot convert to integer
    # b = 1.5
    # c_add = a + b
    # print(c_add)
    # self.assertEqual(c_add.dtype, core.VarDesc.VarType.FP32)

    # AssertionError: VarType.INT32 != VarType.FP32
    # b = 2.0
    # c_div = a / b
    # print(c_div)
    # self.assertEqual(c_div.dtype, core.VarDesc.VarType.FP32)

    # def test_float32_promote_to_float64(self):
    #     a = paddle.ones([3, 3, 3], dtype='float32')
    #     b = 1
    #     c_add = a + b
    #     self.assertTrue(np.array_equal(c_add, a + a))
    #     self.assertEqual(c_add.dtype, core.VarDesc.VarType.FP32)

    # def test_int_div_by_int(self):
    #     a = paddle.ones([3, 3, 3], dtype='int32')
    #     b = 2

    # AssertionError: VarType.INT32 != VarType.FP32
    # c_div = a / b
    # print(c_div)
    # self.assertEqual(c_div.dtype, core.VarDesc.VarType.FP32)


if __name__ == '__main__':
    unittest.main()
