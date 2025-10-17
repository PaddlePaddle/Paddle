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

from utils import static_guard

import paddle


class TestPirTensor(unittest.TestCase):
    def test_element_size(self):
        with static_guard():
            x = paddle.to_tensor(1, dtype="bool")
            self.assertEqual(x.itemsize, 1)

            x = paddle.to_tensor(1, dtype="float16")
            self.assertEqual(x.itemsize, 2)

            x = paddle.to_tensor(1, dtype="float32")
            self.assertEqual(x.itemsize, 4)

            x = paddle.to_tensor(1, dtype="float64")
            self.assertEqual(x.itemsize, 8)

            x = paddle.to_tensor(1, dtype="int8")
            self.assertEqual(x.itemsize, 1)

            x = paddle.to_tensor(1, dtype="int16")
            self.assertEqual(x.itemsize, 2)

            x = paddle.to_tensor(1, dtype="int32")
            self.assertEqual(x.itemsize, 4)

            x = paddle.to_tensor(1, dtype="int64")
            self.assertEqual(x.itemsize, 8)

            x = paddle.to_tensor(1, dtype="uint8")
            self.assertEqual(x.itemsize, 1)

            x = paddle.to_tensor(1, dtype="complex64")
            self.assertEqual(x.itemsize, 8)

            x = paddle.to_tensor(1, dtype="complex128")
            self.assertEqual(x.itemsize, 16)


if __name__ == '__main__':
    unittest.main()
