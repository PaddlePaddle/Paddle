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

import paddle
import numpy as np
import unittest


class TestIsInteger(unittest.TestCase):
    def test_for_integer(self):
        x = paddle.arange(10)
        self.assertTrue(paddle.is_integer(x))

    def test_for_floating_point(self):
        x = paddle.randn([2, 3])
        self.assertFalse(paddle.is_integer(x))

    def test_for_complex(self):
        x = paddle.randn([2, 3]) + 1j * paddle.randn([2, 3])
        self.assertFalse(paddle.is_integer(x))

    def test_for_exception(self):
        with self.assertRaises(TypeError):
            paddle.is_integer(np.array([1, 2]))


if __name__ == '__main__':
    unittest.main()
