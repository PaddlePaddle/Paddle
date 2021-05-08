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

from paddle.devices import cuda
import paddle

import unittest


class TestCurrentStream(unittest.TestCase):
    def test_current_stream(self):
        if paddle.is_compiled_with_cuda():
            s = cuda.current_stream()
            self.assertTrue(isinstance(s, cuda.Stream))

            s1 = cuda.current_stream(0)
            self.assertTrue(isinstance(s1, cuda.Stream))

            s2 = cuda.current_stream(paddle.CUDAPlace(0))
            self.assertTrue(isinstance(s2, cuda.Stream))

            self.assertEqual(s1, s2)

            self.assertRaises(ValueError, cuda.current_stream, "gpu:0")


class TestSynchronize(unittest.TestCase):
    def test_synchronize(self):
        if paddle.is_compiled_with_cuda():
            self.assertIsNone(cuda.synchronize())
            self.assertIsNone(cuda.synchronize(0))
            self.assertIsNone(cuda.synchronize(paddle.CUDAPlace(0)))

            self.assertRaises(ValueError, cuda.synchronize, "gpu:0")


if __name__ == "__main__":
    unittest.main()
