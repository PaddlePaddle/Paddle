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

# [AUTO-GENERATED] Unit test for paddle.tensor.stat
# 自动生成的单测，覆盖 paddle.tensor.stat 模块中未覆盖的代码
# Target: cover uncovered lines 237, 566, 777, 873 in paddle/python/paddle/tensor/stat.py
# 目标：覆盖 stat.py 中 var() 参数冲突检查、nanmedian() 类型检查、median() mode 参数校验、quantile() interpolation 参数校验

"""
This test covers the following modules and code paths:
这个测试覆盖以下模块和代码路径：

1. paddle.var() - Error when both 'unbiased' and non-default 'correction' are given (line 237)
   paddle.var() - 当同时提供 'unbiased' 和非默认 'correction' 时的错误处理

2. paddle.nanmedian() - TypeError when input is not a Tensor (line 566)
   paddle.nanmedian() - 输入不是 Tensor 时的类型错误

3. paddle.median() - ValueError when mode is invalid (line 777)
   paddle.median() - mode 参数值无效时的值错误

4. paddle.quantile() - ValueError when interpolation is invalid (line 873)
   paddle.quantile() - interpolation 参数值无效时的值错误
"""

import unittest

import numpy as np

import paddle


class TestVarParameterConflict(unittest.TestCase):
    """Test var() raises ValueError when both unbiased and correction are given.
    测试 var() 在同时传入 unbiased 和 correction 参数时抛出 ValueError。
    覆盖 paddle/python/paddle/tensor/stat.py 第 237 行。
    """

    def test_var_unbiased_and_correction_conflict(self):
        """var() should raise ValueError when both unbiased and non-default correction are given.
        当同时传入 unbiased 参数（非None）和非默认 correction 参数时，var() 应抛出 ValueError。
        """
        x = paddle.to_tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        with self.assertRaises(ValueError):
            paddle.var(x, unbiased=True, correction=2)

    def test_var_unbiased_false_and_correction_conflict(self):
        """var() should also raise ValueError with unbiased=False and correction!=1.
        当 unbiased=False 且 correction 不为默认值 1 时，同样应抛出 ValueError。
        """
        x = paddle.to_tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        with self.assertRaises(ValueError):
            paddle.var(x, unbiased=False, correction=0)

    def test_var_normal_usage(self):
        """Normal var() usage should work correctly.
        正常使用 var() 应返回正确的方差值。
        """
        x = paddle.to_tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result = paddle.var(x)
        np.testing.assert_allclose(
            result.numpy(), np.var([1, 2, 3, 4, 5], ddof=1), rtol=1e-5
        )

    def test_var_with_correction_only(self):
        """var() with correction only (no unbiased) should work.
        仅传入 correction 参数（不传 unbiased）应正常工作。
        """
        x = paddle.to_tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result = paddle.var(x, correction=0)
        np.testing.assert_allclose(
            result.numpy(), np.var([1, 2, 3, 4, 5], ddof=0), rtol=1e-5
        )


class TestNanmedianTypeError(unittest.TestCase):
    """Test nanmedian() raises TypeError when input is not a Tensor.
    测试 nanmedian() 在输入不是 Tensor 时抛出 TypeError。
    覆盖 paddle/python/paddle/tensor/stat.py 第 566 行。
    """

    def test_nanmedian_with_int_input(self):
        """nanmedian() should raise TypeError for integer input.
        传入整数时 nanmedian() 应抛出 TypeError。
        """
        with self.assertRaises(TypeError):
            paddle.nanmedian(123)

    def test_nanmedian_with_list_input(self):
        """nanmedian() should raise TypeError for list input.
        传入列表时 nanmedian() 应抛出 TypeError。
        """
        with self.assertRaises(TypeError):
            paddle.nanmedian([1.0, 2.0, 3.0])

    def test_nanmedian_with_numpy_input(self):
        """nanmedian() should raise TypeError for numpy array input.
        传入 numpy 数组时 nanmedian() 应抛出 TypeError。
        """
        with self.assertRaises(TypeError):
            paddle.nanmedian(np.array([1.0, 2.0, 3.0]))

    def test_nanmedian_normal_usage(self):
        """Normal nanmedian() usage with Tensor should work.
        正常传入 Tensor 时 nanmedian() 应正常工作。
        """
        x = paddle.to_tensor([1.0, float('nan'), 3.0, 2.0])
        result = paddle.nanmedian(x)
        np.testing.assert_allclose(result.numpy(), 2.0, rtol=1e-5)


class TestMedianInvalidMode(unittest.TestCase):
    """Test median() raises ValueError when mode is invalid.
    测试 median() 在 mode 参数无效时抛出 ValueError。
    覆盖 paddle/python/paddle/tensor/stat.py 第 777 行。
    """

    def test_median_invalid_mode(self):
        """median() should raise ValueError for mode not in ['avg', 'min'].
        当 mode 不是 'avg' 或 'min' 时应抛出 ValueError。
        """
        x = paddle.to_tensor([1.0, 2.0, 3.0, 4.0])
        with self.assertRaises(ValueError):
            paddle.median(x, mode='max')

    def test_median_invalid_mode_string(self):
        """median() should raise ValueError for arbitrary string mode.
        传入任意字符串 mode 值时应抛出 ValueError。
        """
        x = paddle.to_tensor([1.0, 2.0, 3.0, 4.0])
        with self.assertRaises(ValueError):
            paddle.median(x, mode='invalid')

    def test_median_valid_mode_avg(self):
        """median() with mode='avg' should work correctly.
        mode='avg' 应正常工作。
        """
        x = paddle.to_tensor([1.0, 2.0, 3.0, 4.0])
        result = paddle.median(x, mode='avg')
        np.testing.assert_allclose(result.numpy(), 2.5, rtol=1e-5)

    def test_median_valid_mode_min(self):
        """median() with mode='min' should work correctly.
        mode='min' 应正常工作。
        """
        x = paddle.to_tensor([1.0, 2.0, 3.0, 4.0])
        result = paddle.median(x, mode='min')
        np.testing.assert_allclose(result.numpy(), 2.0, rtol=1e-5)


class TestQuantileInvalidInterpolation(unittest.TestCase):
    """Test quantile() raises ValueError when interpolation is invalid.
    测试 quantile() 在 interpolation 参数无效时抛出 ValueError。
    覆盖 paddle/python/paddle/tensor/stat.py 第 873 行。
    """

    def test_quantile_invalid_interpolation(self):
        """quantile() should raise ValueError for invalid interpolation.
        传入无效 interpolation 参数时应抛出 ValueError。
        """
        x = paddle.to_tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        with self.assertRaises(ValueError):
            paddle.quantile(x, q=0.5, interpolation='invalid')

    def test_quantile_invalid_interpolation_empty(self):
        """quantile() should raise ValueError for empty string interpolation.
        传入空字符串 interpolation 参数时应抛出 ValueError。
        """
        x = paddle.to_tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        with self.assertRaises(ValueError):
            paddle.quantile(x, q=0.5, interpolation='')

    def test_quantile_valid_interpolation_linear(self):
        """quantile() with interpolation='linear' should work.
        interpolation='linear' 应正常工作。
        """
        x = paddle.to_tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result = paddle.quantile(x, q=0.5, interpolation='linear')
        np.testing.assert_allclose(result.numpy(), 3.0, rtol=1e-5)

    def test_quantile_valid_interpolation_lower(self):
        """quantile() with interpolation='lower' should work.
        interpolation='lower' 应正常工作。
        """
        x = paddle.to_tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result = paddle.quantile(x, q=0.25, interpolation='lower')
        np.testing.assert_allclose(result.numpy(), 2.0, rtol=1e-5)

    def test_nanquantile_invalid_interpolation(self):
        """nanquantile() should also raise ValueError for invalid interpolation.
        nanquantile() 同样应在 interpolation 无效时抛出 ValueError。
        """
        x = paddle.to_tensor([1.0, float('nan'), 3.0, 4.0, 5.0])
        with self.assertRaises(ValueError):
            paddle.nanquantile(x, q=0.5, interpolation='cubic')


if __name__ == '__main__':
    unittest.main()
