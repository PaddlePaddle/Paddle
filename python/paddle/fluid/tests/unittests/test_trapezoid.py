# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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


class TestTrapezoid(unittest.TestCase):
    def setUp(self):
        # set up some test cases
        self.x1 = paddle.to_tensor([[1, 2, 3], [3, 4, 5]])
        self.y1 = paddle.to_tensor([[2, 4, 8], [3, 5, 9]])
        self.x2 = paddle.to_tensor([1.0])
        self.y2 = paddle.to_tensor([2.0])
        self.x3 = paddle.to_tensor([1.0, float('inf')])
        self.y3 = paddle.to_tensor([2.0, float('nan')])

    def test_trapezoid(self):
        # test normal case
        result = paddle.trapezoid(self.y1, self.x1, axis=-1)
        expected = paddle.to_tensor([9.0, 11.0])
        self.assertTrue(paddle.allclose(result, expected))

    def test_trapezoid_dx(self):
        # test dx argument
        result = paddle.trapezoid(self.y1)
        expected = paddle.to_tensor([9.0, 11.0])
        self.assertTrue(paddle.allclose(result, expected))

    def test_trapezoid_axis(self):
        # test axis argument
        result = paddle.trapezoid(self.y1, axis=0)
        expected = paddle.to_tensor([2.5, 4.5, 8.5])

        self.assertTrue(paddle.allclose(result, expected))

    def test_trapezoid_type_error(self):
        # test type error
        with self.assertRaises(TypeError):
            paddle.trapezoid(1, self.x1)  # y is not a tensor
        with self.assertRaises(TypeError):
            paddle.trapezoid(self.y1, "a")  # x is not a tensor

    def test_trapezoid_value_error(self):
        # test value error
        with self.assertRaises(ValueError):
            paddle.trapezoid(
                self.y3, self.x3
            )  # y and x contain inf or nan values
        with self.assertRaises(ValueError):
            paddle.trapezoid(self.y2, self.x2, dx=0)  # dx is not positive


if __name__ == "__main__":
    unittest.main()
