# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import paddle


class TensorFloat16Test(unittest.TestCase):
    def test_float16(self):
        value = np.array([1.5, 2.5, 3.5]).astype("float32")
        tensor = paddle.to_tensor(value, dtype="float32")

        self.assertEqual(tensor.dtype, paddle.float32)

        float16_tensor = tensor.float16()
        self.assertEqual(float16_tensor.dtype, paddle.float16)


class TensorBfloat16Test(unittest.TestCase):
    def test_float16(self):
        value = np.array([1.5, 2.5, 3.5]).astype("float32")
        tensor = paddle.to_tensor(value, dtype="float32")

        self.assertEqual(tensor.dtype, paddle.float32)

        float16_tensor = tensor.bfloat16()
        self.assertEqual(float16_tensor.dtype, paddle.bfloat16)


if __name__ == '__main__':
    unittest.main()
