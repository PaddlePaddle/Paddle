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

class TestListToTensor(unittest.TestCase):
    def test_list_to_tensor_bfloat16(self):
        a = [paddle.to_tensor(2,dtype=paddle.bfloat16)]
        b = paddle.to_tensor(a)
        self.assertEqual(b.dtype, paddle.bfloat16)
        self.assertEqual(b[0], 2.0)
    def test_list_to_tensor_float16(self):
        a = [paddle.to_tensor(2,dtype=paddle.float16)]
        b = paddle.to_tensor(a)
        self.assertEqual(b.dtype, paddle.float16)
        self.assertEqual(b[0], 2.0)

    def test_list_to_tensor_bfloat16_float32(self):
        a = [paddle.to_tensor(2,dtype=paddle.bfloat16), paddle.to_tensor(2,dtype=paddle.float32)]
        b = paddle.to_tensor(a)
        self.assertEqual(b.dtype, paddle.float32)
        self.assertEqual(b[0], 2.0)
        self.assertEqual(b[1], 2.0)

    def test_list_to_tensor_float16_float32(self):
        a = [paddle.to_tensor(2,dtype=paddle.float16), paddle.to_tensor(2,dtype=paddle.float32)]
        b = paddle.to_tensor(a)
        self.assertEqual(b.dtype, paddle.float32)
        self.assertEqual(b[0], 2.0)
        self.assertEqual(b[1], 2.0)

if __name__ == '__main__':
    unittest.main()
