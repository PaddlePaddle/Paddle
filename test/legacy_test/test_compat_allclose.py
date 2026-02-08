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

import paddle
from paddle.compat import allclose


class TestCompatAllclose(unittest.TestCase):
    def setUp(self):
        self.places = [paddle.CPUPlace()]
        if paddle.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

    def test_allclose_return_bool(self):
        """Test compat.allclose returns python bool"""
        for place in self.places:
            with self.subTest(place=place):
                x = paddle.to_tensor([10000.0, 1e-07])
                y = paddle.to_tensor([10000.1, 1e-08])

                # Test return type
                res = allclose(x, y)
                self.assertIsInstance(res, bool)
                self.assertFalse(res)

                # Test True case
                x2 = paddle.to_tensor([1.0, 2.0])
                y2 = paddle.to_tensor([1.0, 2.0])
                res2 = allclose(x2, y2)
                self.assertIsInstance(res2, bool)
                self.assertTrue(res2)

    def test_allclose_args(self):
        """Test compat.allclose arguments"""
        for place in self.places:
            with self.subTest(place=place):
                x = paddle.to_tensor([1.0, float('nan')])
                y = paddle.to_tensor([1.0, float('nan')])

                # Test equal_nan
                res = allclose(input=x, other=y, equal_nan=True)
                self.assertTrue(res)

                res = allclose(x, y, equal_nan=False)
                self.assertFalse(res)


if __name__ == "__main__":
    unittest.main()
