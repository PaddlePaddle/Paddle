# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from paddle.base import core


class Float8_E4M3_Test(unittest.TestCase):
    def setUp(self):
        self.dtype = "float8_e4m3"
        self.paddle_dtype = core.VarDesc.VarType.FP8_E4M3

    def test_fullOp(self):
        input1 = paddle.ones([2, 3], dtype=self.dtype)
        self.assertTrue(input1.dtype == self.paddle_dtype)

    def test_castOp(self):
        input1 = paddle.ones([2, 3])
        input1 = input1.astype(self.dtype)
        self.assertTrue(input1.dtype == self.paddle_dtype)


class Float8_E5M2_Test(Float8_E4M3_Test):
    def setUp(self):
        self.dtype = "float8_e5m2"
        self.paddle_dtype = core.VarDesc.VarType.FP8_E5M2


if __name__ == "__main__":
    unittest.main()
