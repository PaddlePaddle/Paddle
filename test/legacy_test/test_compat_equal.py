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
from paddle.compat import equal


class TestCompatEqual(unittest.TestCase):
    def test_equal_tensors(self):
        """Test equal tensors return True"""
        x = paddle.to_tensor([1.0, 2.0, 3.0])
        y = paddle.to_tensor([1.0, 2.0, 3.0])
        self.assertTrue(equal(x, y))

        x_int = paddle.to_tensor([1, 2, 3], dtype='int32')
        y_int = paddle.to_tensor([1, 2, 3], dtype='int32')
        self.assertTrue(equal(x_int, y_int))

    def test_unequal_tensors(self):
        """Test unequal tensors return False"""
        x = paddle.to_tensor([1.0, 2.0, 3.0])
        y = paddle.to_tensor([1.0, 2.0, 4.0])
        self.assertFalse(equal(x, y))

        x = paddle.to_tensor([1.0, 2.0, 3.0])
        y = paddle.to_tensor([4.0, 5.0, 6.0])
        self.assertFalse(equal(x, y))

    def test_different_dtypes(self):
        """Test tensors with different dtypes"""
        x_float = paddle.to_tensor([1.0, 2.0, 3.0], dtype='float32')
        y_int = paddle.to_tensor([1, 2, 3], dtype='float64')
        self.assertTrue(equal(x_float, y_int))

    def test_different_ndim(self):
        """Test tensors with different number of dimensions"""
        x_1d = paddle.to_tensor([1.0, 2.0, 3.0])
        x_2d = paddle.to_tensor([[1.0, 2.0, 3.0]])
        self.assertFalse(equal(x_1d, x_2d))

    def test_different_shapes(self):
        """Test tensors with same ndim but different shapes"""
        x = paddle.to_tensor([[1.0, 2.0, 3.0]])
        y = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
        self.assertFalse(equal(x, y))

        x = paddle.rand([2, 3, 4])
        y = paddle.rand([2, 4, 3])
        self.assertFalse(equal(x, y))

    def test_empty_tensors(self):
        """Test empty tensors"""
        x_empty = paddle.to_tensor([], dtype='float32')
        y_empty = paddle.to_tensor([], dtype='float32')
        self.assertTrue(equal(x_empty, y_empty))

        x_empty_1d = paddle.to_tensor([], dtype='float32')
        y_empty_2d = paddle.to_tensor([[]], dtype='float32')
        self.assertFalse(equal(x_empty_1d, y_empty_2d))

    def test_broadcast_shapes(self):
        """Test tensors that could be broadcast but have different shapes"""
        x = paddle.to_tensor([1.0, 2.0, 3.0])
        y = paddle.to_tensor([[1.0, 2.0, 3.0]])
        self.assertFalse(equal(x, y))

    def test_complex_tensors(self):
        """Test with complex tensor structures"""
        x = paddle.arange(24).reshape([2, 3, 4]).astype('float32')
        y = paddle.arange(24).reshape([2, 3, 4]).astype('float32')
        self.assertTrue(equal(x, y))

        z = x.clone()
        z[0, 0, 0] = 100.0
        self.assertFalse(equal(x, z))

    def test_nan_and_inf(self):
        """Test with NaN and Inf values"""
        x_nan = paddle.to_tensor([1.0, float('nan'), 3.0])
        y_nan = paddle.to_tensor([1.0, float('nan'), 3.0])
        self.assertFalse(equal(x_nan, y_nan))

        x_inf = paddle.to_tensor([1.0, float('inf'), 3.0])
        y_inf = paddle.to_tensor([1.0, float('inf'), 3.0])
        self.assertTrue(equal(x_inf, y_inf))

        x_neg_inf = paddle.to_tensor([1.0, float('-inf'), 3.0])
        y_neg_inf = paddle.to_tensor([1.0, float('-inf'), 3.0])
        self.assertTrue(equal(x_neg_inf, y_neg_inf))

    def test_very_large_tensors(self):
        """Test with very large tensors"""
        x_large = paddle.ones([100, 100])
        y_large = paddle.ones([100, 100])
        self.assertTrue(equal(x_large, y_large))

        z_large = x_large.clone()
        z_large[50, 50] = 2.0
        self.assertFalse(equal(x_large, z_large))

    def test_error_cases(self):
        """Test error handling"""
        with self.assertRaises(AttributeError):
            equal([1, 2, 3], paddle.to_tensor([1, 2, 3]))

        with self.assertRaises(AttributeError):
            equal(paddle.to_tensor([1, 2, 3]), [1, 2, 3])


if __name__ == '__main__':
    unittest.main()
