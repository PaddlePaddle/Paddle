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

# test_cuda_unittest.py
import unittest

import paddle


class TestCudaCompat(unittest.TestCase):
    # ---------------------
    # paddle.device compatibility tests
    # ---------------------
    def test_paddle_device_cpu(self):
        d = paddle.device("cpu")
        self.assertTrue(d == "cpu")

    def test_paddle_device_gpu(self):
        d1 = paddle.device("cuda", 2)
        self.assertEqual(d1, "gpu:2")

        d2 = paddle.device("cuda:3")
        self.assertEqual(d2, "gpu:3")

        d3 = paddle.device(4)
        self.assertEqual(d3, "gpu:4")

    def test_paddle_device_copy(self):
        d1 = paddle.device("gpu:1")
        d2 = paddle.device(d1)
        self.assertEqual(d1, d2)

    def test_paddle_device_invalid(self):
        with self.assertRaises(ValueError):
            paddle.device("tpu")
        with self.assertRaises(TypeError):
            paddle.device(3.14)


if __name__ == '__main__':
    unittest.main()
