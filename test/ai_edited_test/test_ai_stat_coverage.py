# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

# [AUTO-GENERATED] Tests for paddle/tensor/stat.py (coverage: 81.2% -> higher)
# Target file: python/paddle/tensor/stat.py
# Functions: mean, var, std, numel, nanmedian, median, quantile, nanquantile

import unittest

import numpy as np

import paddle


class TestMeanBasic(unittest.TestCase):
    """测试 mean 基本功能 / Test basic mean functionality."""

    def setUp(self):
        self.x = paddle.to_tensor(
            [
                [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
                [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]],
            ],
            dtype='float32',
        )

    def tearDown(self):
        pass

    def test_mean_all(self):
        """测试全元素均值 / Test mean over all elements."""
        out = paddle.mean(self.x)
        np.testing.assert_allclose(out.numpy(), 12.5, rtol=1e-5)

    def test_mean_axis_neg1(self):
        """测试沿 axis=-1 均值 / Test mean along axis=-1."""
        out = paddle.mean(self.x, axis=-1)
        self.assertEqual(out.shape, [2, 3])

    def test_mean_keepdim(self):
        """测试 keepdim 均值 / Test mean with keepdim."""
        out = paddle.mean(self.x, axis=-1, keepdim=True)
        self.assertEqual(out.shape, [2, 3, 1])

    def test_mean_multi_axis(self):
        """测试多轴均值 / Test mean along multiple axes."""
        out = paddle.mean(self.x, axis=[0, 2])
        self.assertEqual(out.shape, [3])

    def test_mean_dtype(self):
        """测试指定输出 dtype / Test mean with specified dtype."""
        out = paddle.mean(self.x, dtype='float64')
        self.assertEqual(out.dtype, paddle.float64)

    def test_mean_out(self):
        """测试 mean 带 out 参数 / Test mean with out parameter."""
        out = paddle.empty([3], dtype='float32')
        paddle.mean(self.x, axis=[0, 2], out=out)
        self.assertEqual(out.shape, [3])


class TestVarBasic(unittest.TestCase):
    """测试 var 基本功能 / Test basic var functionality."""

    def test_var_all(self):
        """测试全元素方差 / Test variance over all elements."""
        x = paddle.to_tensor([[1, 2, 3], [1, 4, 5]], dtype='float32')
        out = paddle.var(x)
        np.testing.assert_allclose(out.numpy(), 2.6666667, rtol=1e-4)

    def test_var_axis(self):
        """测试沿轴方差 / Test variance along axis."""
        x = paddle.to_tensor([[1, 2, 3], [1, 4, 5]], dtype='float32')
        out = paddle.var(x, axis=1)
        self.assertEqual(out.shape, [2])

    def test_var_biased(self):
        """测试有偏方差 / Test biased variance."""
        x = paddle.to_tensor([[1, 2, 3], [1, 4, 5]], dtype='float32')
        out = paddle.var(x, unbiased=False)
        self.assertTrue(out.numpy() < paddle.var(x).numpy())

    def test_var_correction(self):
        """测试 correction 参数 / Test correction parameter."""
        x = paddle.to_tensor([[1, 2, 3], [1, 4, 5]], dtype='float32')
        out = paddle.var(x, unbiased=False)
        # Biased variance: sum of squared deviations / N
        self.assertAlmostEqual(float(out.numpy()), 2.222222, places=4)

    def test_var_unbiased_correction_conflict(self):
        """测试 unbiased 和 correction 同时传参报错 / Test unbiased and correction conflict."""
        x = paddle.to_tensor([[1, 2, 3]], dtype='float32')
        with self.assertRaises(ValueError):
            paddle.var(x, unbiased=True, correction=0.5)

    def test_var_empty_dim(self):
        """测试空维度方差 / Test variance with empty dimension."""
        x = paddle.to_tensor([], dtype='float32').reshape([0, 3])
        out = paddle.var(x)
        self.assertTrue(paddle.isnan(out).numpy())

    def test_var_zero_dim_biased(self):
        """测试 0D 张量有偏方差 / Test 0D tensor biased variance."""
        x = paddle.to_tensor(5.0, dtype='float32')
        out = paddle.var(x, correction=0.0)
        np.testing.assert_allclose(out.numpy(), 0.0, rtol=1e-5)

    def test_var_keepdim(self):
        """测试方差 keepdim / Test variance with keepdim."""
        x = paddle.to_tensor([[1, 2, 3], [4, 5, 6]], dtype='float32')
        out = paddle.var(x, axis=1, keepdim=True)
        self.assertEqual(out.shape, [2, 1])

    def test_var_out(self):
        """测试 var 带 out 参数 / Test var with out parameter."""
        x = paddle.to_tensor([[1, 2, 3]], dtype='float32')
        out = paddle.empty([], dtype='float32')
        result = paddle.var(x, out=out)
        self.assertEqual(result.shape, [])


class TestStdBasic(unittest.TestCase):
    """测试 std 基本功能 / Test basic std functionality."""

    def test_std_all(self):
        """测试全元素标准差 / Test std over all elements."""
        x = paddle.to_tensor([[1, 2, 3], [1, 4, 5]], dtype='float32')
        out = paddle.std(x)
        np.testing.assert_allclose(out.numpy(), 1.6329932, rtol=1e-4)

    def test_std_biased(self):
        """测试有偏标准差 / Test biased std."""
        x = paddle.to_tensor([[1, 2, 3], [1, 4, 5]], dtype='float32')
        out = paddle.std(x, unbiased=False)
        self.assertTrue(out.numpy() < paddle.std(x).numpy())

    def test_std_axis(self):
        """测试沿轴标准差 / Test std along axis."""
        x = paddle.to_tensor([[1, 2, 3], [1, 4, 5]], dtype='float32')
        out = paddle.std(x, axis=1)
        self.assertEqual(out.shape, [2])

    def test_std_keepdim(self):
        """测试标准差 keepdim / Test std with keepdim."""
        x = paddle.to_tensor([[1, 2, 3], [1, 4, 5]], dtype='float32')
        out = paddle.std(x, keepdim=True)
        self.assertEqual(out.shape, [1, 1])

    def test_std_correction(self):
        """测试 correction 参数 / Test std with correction parameter."""
        x = paddle.to_tensor([[1, 2, 3], [1, 4, 5]], dtype='float32')
        out = paddle.std(x, correction=1.5)
        self.assertEqual(out.shape, [])

    def test_std_dim_alias(self):
        """测试 dim 别名 / Test std with dim alias."""
        x = paddle.to_tensor([[1, 2, 3], [1, 4, 5]], dtype='float32')
        out = paddle.std(input=x, dim=[0, 1])
        self.assertEqual(out.shape, [])

    def test_std_out(self):
        """测试 std 带 out 参数 / Test std with out parameter."""
        x = paddle.to_tensor([[1, 2, 3]], dtype='float32')
        out = paddle.empty([], dtype='float32')
        result = paddle.std(x, out=out)
        self.assertEqual(result.shape, [])


class TestNumel(unittest.TestCase):
    """测试 numel 功能 / Test numel functionality."""

    def test_numel_basic(self):
        """测试 numel 基本功能 / Test basic numel."""
        x = paddle.full(shape=[4, 5, 7], fill_value=0, dtype='int32')
        out = paddle.numel(x)
        np.testing.assert_equal(out.numpy(), 140)

    def test_numel_1d(self):
        """测试 1D numel / Test 1D numel."""
        x = paddle.zeros([10])
        out = paddle.numel(x)
        np.testing.assert_equal(out.numpy(), 10)

    def test_numel_scalar(self):
        """测试标量 numel / Test scalar numel."""
        x = paddle.to_tensor(5.0)
        out = paddle.numel(x)
        np.testing.assert_equal(out.numpy(), 1)


class TestNanmedian(unittest.TestCase):
    """测试 nanmedian 功能 / Test nanmedian functionality."""

    def test_nanmedian_basic(self):
        """测试 nanmedian 基本功能 / Test basic nanmedian."""
        x = paddle.to_tensor(
            [[float('nan'), 2.0, 3.0], [0.0, 1.0, 2.0]], dtype='float32'
        )
        out = x.nanmedian()
        np.testing.assert_allclose(out.numpy(), 2.0, rtol=1e-5)

    def test_nanmedian_axis(self):
        """测试 nanmedian 沿轴 / Test nanmedian along axis."""
        x = paddle.to_tensor(
            [[float('nan'), 2.0, 3.0], [0.0, 1.0, 2.0]], dtype='float32'
        )
        out = x.nanmedian(0)
        self.assertEqual(out.shape, [3])

    def test_nanmedian_keepdim(self):
        """测试 nanmedian keepdim / Test nanmedian with keepdim."""
        x = paddle.to_tensor(
            [[float('nan'), 2.0, 3.0], [0.0, 1.0, 2.0]], dtype='float32'
        )
        out = x.nanmedian(0, keepdim=True)
        self.assertEqual(out.shape, [1, 3])

    def test_nanmedian_multi_axis(self):
        """测试 nanmedian 多轴 / Test nanmedian with multiple axes."""
        x = paddle.to_tensor(
            [[float('nan'), 2.0, 3.0], [0.0, 1.0, 2.0]], dtype='float32'
        )
        out = x.nanmedian((0, 1))
        self.assertEqual(out.shape, [])

    def test_nanmedian_mode_min(self):
        """测试 nanmedian mode='min' / Test nanmedian with mode='min'."""
        x = paddle.to_tensor(
            [[float('nan'), 2.0, 3.0], [0.0, 1.0, 2.0]], dtype='float32'
        )
        out = x.nanmedian(mode='min')
        np.testing.assert_allclose(out.numpy(), 2.0, rtol=1e-5)

    def test_nanmedian_axis_mode_min(self):
        """测试 nanmedian 沿轴 mode='min' / Test nanmedian with axis and mode='min'."""
        x = paddle.to_tensor(
            [[float('nan'), 2.0, 3.0], [0.0, 1.0, 2.0]], dtype='float32'
        )
        out, idx = x.nanmedian(0, mode='min')
        self.assertEqual(out.shape, [3])
        self.assertEqual(idx.shape, [3])


class TestMedian(unittest.TestCase):
    """测试 median 功能 / Test median functionality."""

    def test_median_basic(self):
        """测试 median 基本功能 / Test basic median."""
        x = paddle.arange(12, dtype='int64').reshape([3, 4])
        out = paddle.median(x)
        np.testing.assert_allclose(out.numpy(), 5.5, rtol=1e-5)

    def test_median_axis(self):
        """测试 median 沿轴 / Test median along axis."""
        x = paddle.arange(12, dtype='int64').reshape([3, 4])
        out = paddle.median(x, axis=0)
        self.assertEqual(out.shape, [4])

    def test_median_keepdim(self):
        """测试 median keepdim / Test median with keepdim."""
        x = paddle.arange(12, dtype='int64').reshape([3, 4])
        out = paddle.median(x, axis=0, keepdim=True)
        self.assertEqual(out.shape, [1, 4])

    def test_median_mode_min(self):
        """测试 median mode='min' / Test median with mode='min'."""
        x = paddle.arange(12, dtype='int64').reshape([3, 4])
        out = paddle.median(x, mode='min')
        self.assertEqual(out.shape, [])

    def test_median_axis_mode_min(self):
        """测试 median 沿轴 mode='min' / Test median with axis and mode='min'."""
        x = paddle.arange(12, dtype='int64').reshape([3, 4])
        values, indices = paddle.median(x, axis=1, mode='min')
        self.assertEqual(values.shape, [3])
        self.assertEqual(indices.shape, [3])

    def test_median_dim_alias(self):
        """测试 dim 别名 / Test median with dim alias."""
        x = paddle.arange(12, dtype='int64').reshape([3, 4])
        out = paddle.median(input=x, dim=0)
        # With dim alias, mode defaults to 'min', so returns tuple
        if isinstance(out, tuple):
            self.assertEqual(out[0].shape, [4])
        else:
            self.assertEqual(out.shape, [4])

    def test_median_type_error(self):
        """测试非张量输入报错 / Test non-tensor input raises error."""
        with self.assertRaises(TypeError):
            paddle.median([1, 2, 3])


class TestQuantile(unittest.TestCase):
    """测试 quantile 功能 / Test quantile functionality."""

    def test_quantile_basic(self):
        """测试 quantile 基本功能 / Test basic quantile."""
        x = paddle.arange(0, 8, dtype='float32').reshape([4, 2])
        out = paddle.quantile(x, q=0.5, axis=[0, 1])
        np.testing.assert_allclose(out.numpy(), 3.5, rtol=1e-5)

    def test_quantile_axis(self):
        """测试 quantile 沿轴 / Test quantile along axis."""
        x = paddle.arange(0, 8, dtype='float32').reshape([4, 2])
        out = paddle.quantile(x, q=0.5, axis=1)
        self.assertEqual(out.shape, [4])

    def test_quantile_list_q(self):
        """测试 quantile 多个 q / Test quantile with list of q values."""
        x = paddle.arange(0, 8, dtype='float32').reshape([4, 2])
        out = paddle.quantile(x, q=[0.3, 0.5], axis=0)
        self.assertEqual(out.shape, [2, 2])

    def test_quantile_keepdim(self):
        """测试 quantile keepdim / Test quantile with keepdim."""
        x = paddle.arange(0, 8, dtype='float32').reshape([4, 2])
        out = paddle.quantile(x, q=0.8, axis=1, keepdim=True)
        self.assertEqual(out.shape, [4, 1])

    def test_quantile_single_q(self):
        """测试 quantile 标量 q / Test quantile with scalar q."""
        x = paddle.arange(0, 8, dtype='float32').reshape([4, 2])
        out = paddle.quantile(x, q=0.25)
        self.assertEqual(out.shape, [])

    def test_quantile_type_error(self):
        """测试非张量输入报错 / Test non-tensor input raises error."""
        with self.assertRaises(TypeError):
            paddle.quantile([1, 2, 3], q=0.5)


class TestNanquantile(unittest.TestCase):
    """测试 nanquantile 功能 / Test nanquantile functionality."""

    def test_nanquantile_basic(self):
        """测试 nanquantile 基本功能 / Test basic nanquantile."""
        x = paddle.to_tensor(
            [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype='float32'
        )
        x[0, 0] = float('nan')
        out = paddle.nanquantile(x, q=0.5, axis=[0, 1])
        np.testing.assert_allclose(out.numpy(), 5.0, rtol=1e-5)

    def test_nanquantile_axis(self):
        """测试 nanquantile 沿轴 / Test nanquantile along axis."""
        x = paddle.to_tensor(
            [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype='float32'
        )
        x[0, 0] = float('nan')
        out = paddle.nanquantile(x, q=0.5, axis=1)
        self.assertEqual(out.shape, [2])

    def test_nanquantile_list_q(self):
        """测试 nanquantile 多个 q / Test nanquantile with list of q values."""
        x = paddle.to_tensor(
            [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype='float32'
        )
        x[0, 0] = float('nan')
        out = paddle.nanquantile(x, q=[0.3, 0.5], axis=0)
        self.assertEqual(out.shape, [2, 5])

    def test_nanquantile_keepdim(self):
        """测试 nanquantile keepdim / Test nanquantile with keepdim."""
        x = paddle.to_tensor(
            [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype='float32'
        )
        x[0, 0] = float('nan')
        out = paddle.nanquantile(x, q=0.8, axis=1, keepdim=True)
        self.assertEqual(out.shape, [2, 1])

    def test_nanquantile_all_nan(self):
        """测试全 NaN 输入 / Test all-NaN input."""
        nan = paddle.full([2, 3], float('nan'))
        out = paddle.nanquantile(nan, q=0.8, axis=1, keepdim=True)
        self.assertEqual(out.shape, [2, 1])
        self.assertTrue(paddle.isnan(out[0]).numpy())

    def test_nanquantile_dim_alias(self):
        """测试 dim 别名 / Test nanquantile with dim alias."""
        x = paddle.to_tensor([[0, 1, 2], [5, 6, 7]], dtype='float32')
        out = paddle.nanquantile(input=x, dim=0, q=0.5)
        self.assertEqual(out.shape, [3])


if __name__ == '__main__':
    unittest.main()
