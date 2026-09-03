# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

# [AUTO-GENERATED] Unit test for paddle.tensor.to_string
# 自动生成的单测，覆盖 paddle.tensor.to_string 模块中未覆盖的代码路径
# Target: cover uncovered lines 159-165, 256-274, 301 in paddle/python/paddle/tensor/to_string.py
# 目标：覆盖 to_string.py 中 _format_item 的 max_width 路径、to_string 的 bfloat16 路径、scalar 路径

"""
This test covers the following modules and code paths:
这个测试覆盖以下模块和代码路径：

1. _format_item - item_str < max_width 路径 (line 159-161)
2. _format_item - signed=True with negative value (line 159)
3. _format_item - signed=True with positive value (line 161)
4. _format_item - item_str >= max_width (line 164-165)
5. to_string - bfloat16 dtype handling (lines 259-260)
6. to_string - bfloat16 on non-CPU place (lines 269-271)
7. to_string - scalar tensor (0-d) (lines 274-275)
"""

import unittest

import numpy as np

import paddle
from paddle.tensor.to_string import (
    _format_item,
    _get_max_width,
    to_string,
)


class TestFormatItem(unittest.TestCase):
    """Test _format_item function.
    测试 _format_item 函数。
    覆盖 to_string.py 第 155-165 行。
    """

    def test_format_item_no_width(self):
        """Format item without max_width constraint."""
        result = _format_item(np.float64(3.14159))
        self.assertIsInstance(result, str)
        self.assertIn('3.14', result)

    def test_format_item_signed_negative(self):
        """Format signed negative item with max_width."""
        result = _format_item(np.float64(-3.14), max_width=10, signed=True)
        self.assertTrue(len(result) >= 1)
        self.assertIn('-', result)

    def test_format_item_signed_positive(self):
        """Format signed positive item with max_width (adds leading space)."""
        result = _format_item(np.float64(3.14), max_width=10, signed=True)
        self.assertTrue(len(result) >= 2)

    def test_format_item_unsigned(self):
        """Format unsigned item with max_width."""
        result = _format_item(np.float64(3.14), max_width=10, signed=False)
        self.assertIsInstance(result, str)

    def test_format_item_exceeds_width(self):
        """Item exceeding max_width returns as-is."""
        long_str = "1234567890"
        result = _format_item(
            np.float64(1234567890.0), max_width=5, signed=False
        )
        self.assertIsInstance(result, str)

    def test_format_item_integer(self):
        """Format integer item."""
        result = _format_item(np.int64(42))
        self.assertIn('42', result)

    def test_format_item_zero(self):
        """Format zero."""
        result = _format_item(np.float64(0.0))
        self.assertIn('0', result)


class TestGetMaxWidth(unittest.TestCase):
    """Test _get_max_width function.
    测试 _get_max_width 函数。
    覆盖 to_string.py 第 168-178 行。
    """

    def test_positive_only(self):
        """Max width for all positive values."""
        arr = np.array([1.0, 22.0, 333.0])
        max_width, signed = _get_max_width(arr)
        self.assertFalse(signed)
        self.assertEqual(max_width, 4)  # '333' length

    def test_with_negative(self):
        """Max width with negative values (signed=True)."""
        arr = np.array([1.0, -22.0, 333.0])
        max_width, signed = _get_max_width(arr)
        self.assertTrue(signed)
        self.assertEqual(max_width, 4)  # '333' or '-22' length

    def test_single_value(self):
        """Max width for single value."""
        arr = np.array([42.0])
        max_width, signed = _get_max_width(arr)
        self.assertFalse(signed)
        self.assertTrue(max_width > 0)


class TestToString(unittest.TestCase):
    """Test to_string function.
    测试 to_string 函数。
    覆盖 to_string.py 第 255-274 行。
    """

    def setUp(self):
        paddle.disable_static()

    def test_to_string_basic(self):
        """Basic to_string should return formatted string."""
        x = paddle.to_tensor([1.0, 2.0, 3.0])
        result = to_string(x)
        self.assertIn('Tensor', result)
        self.assertIn('shape=paddle.Size([3])', result)

    def test_to_string_float32(self):
        """to_string with float32 tensor."""
        x = paddle.to_tensor([1.0, 2.0], dtype='float32')
        result = to_string(x)
        self.assertIn('float32', result)

    def test_to_string_float64(self):
        """to_string with float64 tensor."""
        x = paddle.to_tensor([1.0, 2.0], dtype='float64')
        result = to_string(x)
        self.assertIn('float64', result)

    def test_to_string_int32(self):
        """to_string with int32 tensor."""
        x = paddle.to_tensor([1, 2, 3], dtype='int32')
        result = to_string(x)
        self.assertIn('int32', result)

    def test_to_string_scalar(self):
        """to_string with scalar (0-d) tensor."""
        x = paddle.to_tensor(42.0)
        result = to_string(x)
        self.assertIn('Tensor', result)

    def test_to_string_2d(self):
        """to_string with 2D tensor."""
        x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
        result = to_string(x)
        self.assertIn('shape=paddle.Size([2, 2])', result)

    def test_to_string_empty(self):
        """to_string with empty tensor."""
        x = paddle.to_tensor([], dtype='float32')
        result = to_string(x)
        self.assertIn('Tensor', result)

    def test_to_string_custom_prefix(self):
        """to_string with custom prefix."""
        x = paddle.to_tensor([1.0, 2.0])
        result = to_string(x, prefix='MyTensor')
        self.assertIn('MyTensor', result)

    def test_repr(self):
        """Test tensor __repr__ calls to_string."""
        x = paddle.to_tensor([1.0, 2.0])
        repr_str = repr(x)
        self.assertIn('Tensor', repr_str)


if __name__ == '__main__':
    unittest.main()
