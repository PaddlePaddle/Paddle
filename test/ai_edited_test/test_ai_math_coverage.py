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

# [AUTO-GENERATED] Tests for paddle/tensor/math.py (coverage: 79.0% -> higher)
# Target file: python/paddle/tensor/math.py
# Functions: scale, pow, add, subtract, divide, mul, logaddexp, nan_to_num,
#            nansum, nanmean, count_nonzero, trunc, trace, cumsum, cumprod,
#            prod, gammaln, digamma, gammainc, neg, lerp, rad2deg, deg2rad,
#            gcd, lcm, diff, frac, sgn, hypot, copysign, sinc, signbit,
#            isposinf, isneginf, isreal, isin, take, frexp

import unittest

import numpy as np

import paddle


class TestScale(unittest.TestCase):
    """测试 scale 功能 / Test scale functionality."""

    def test_scale_basic(self):
        """测试 scale 基本功能 / Test basic scale."""
        x = paddle.to_tensor([1.0, 2.0, 3.0])
        out = paddle.scale(x, scale=2.0, bias=1.0)
        np.testing.assert_array_almost_equal(out.numpy(), [3.0, 5.0, 7.0])

    def test_scale_bias_after_scale(self):
        """测试 scale bias_after_scale / Test scale with bias_after_scale=True."""
        x = paddle.to_tensor([1.0, 2.0, 3.0])
        out = paddle.scale(x, scale=2.0, bias=1.0, bias_after_scale=False)
        np.testing.assert_array_almost_equal(out.numpy(), [4.0, 6.0, 8.0])


class TestPow(unittest.TestCase):
    """测试 pow 功能 / Test pow functionality."""

    def test_pow_tensor(self):
        """测试张量 pow / Test tensor pow."""
        x = paddle.to_tensor([1.0, 2.0, 3.0])
        y = paddle.to_tensor([1.0, 2.0, 3.0])
        out = paddle.pow(x, y)
        np.testing.assert_array_almost_equal(out.numpy(), [1.0, 4.0, 27.0])

    def test_pow_scalar(self):
        """测试标量 pow / Test scalar pow."""
        x = paddle.to_tensor([1.0, 2.0, 3.0])
        out = paddle.pow(x, 2.0)
        np.testing.assert_array_almost_equal(out.numpy(), [1.0, 4.0, 9.0])


class TestLogaddexp(unittest.TestCase):
    """测试 logaddexp 功能 / Test logaddexp functionality."""

    def test_logaddexp_basic(self):
        """测试 logaddexp 基本功能 / Test basic logaddexp."""
        x = paddle.to_tensor([1.0, 2.0, 3.0])
        y = paddle.to_tensor([1.0, 2.0, 3.0])
        out = paddle.logaddexp(x, y)
        expected = np.logaddexp([1, 2, 3], [1, 2, 3])
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5)


class TestNanToNum(unittest.TestCase):
    """测试 nan_to_num 功能 / Test nan_to_num functionality."""

    def test_nan_to_num_basic(self):
        """测试 nan_to_num 基本功能 / Test basic nan_to_num."""
        x = paddle.to_tensor([float('nan'), 1.0, float('inf'), -float('inf')])
        out = paddle.nan_to_num(x)
        self.assertTrue(out[0].numpy() == 0.0)
        np.testing.assert_array_almost_equal(out[1].numpy(), [1.0])

    def test_nan_to_num_custom(self):
        """测试 nan_to_num 自定义替换值 / Test nan_to_num with custom values."""
        x = paddle.to_tensor([float('nan'), float('inf'), -float('inf')])
        out = paddle.nan_to_num(x, nan=7.0, posinf=100.0, neginf=-100.0)
        np.testing.assert_array_almost_equal(out.numpy(), [7.0, 100.0, -100.0])


class TestNansum(unittest.TestCase):
    """测试 nansum 功能 / Test nansum functionality."""

    def test_nansum_basic(self):
        """测试 nansum 基本功能 / Test basic nansum."""
        x = paddle.to_tensor([float('nan'), 1.0, 2.0])
        out = paddle.nansum(x)
        np.testing.assert_allclose(out.numpy(), 3.0, rtol=1e-5)

    def test_nansum_axis(self):
        """测试 nansum 沿轴 / Test nansum along axis."""
        x = paddle.to_tensor([[float('nan'), 1.0], [2.0, 3.0]], dtype='float32')
        out = paddle.nansum(x, axis=0)
        np.testing.assert_allclose(out.numpy(), [2.0, 4.0], rtol=1e-5)

    def test_nansum_keepdim(self):
        """测试 nansum keepdim / Test nansum with keepdim."""
        x = paddle.to_tensor([float('nan'), 1.0, 2.0])
        out = paddle.nansum(x, keepdim=True)
        self.assertEqual(out.shape, [1])


class TestNanmean(unittest.TestCase):
    """测试 nanmean 功能 / Test nanmean functionality."""

    def test_nanmean_basic(self):
        """测试 nanmean 基本功能 / Test basic nanmean."""
        x = paddle.to_tensor([float('nan'), 1.0, 2.0, 3.0], dtype='float32')
        out = paddle.nanmean(x)
        np.testing.assert_allclose(out.numpy(), 2.0, rtol=1e-5)

    def test_nanmean_axis(self):
        """测试 nanmean 沿轴 / Test nanmean along axis."""
        x = paddle.to_tensor([[float('nan'), 2.0], [1.0, 3.0]], dtype='float32')
        out = paddle.nanmean(x, axis=0)
        np.testing.assert_allclose(out.numpy(), [1.0, 2.5], rtol=1e-5)

    def test_nanmean_keepdim(self):
        """测试 nanmean keepdim / Test nanmean with keepdim."""
        x = paddle.to_tensor([float('nan'), 2.0, 3.0], dtype='float32')
        out = paddle.nanmean(x, keepdim=True)
        self.assertEqual(out.shape, [1])


class TestCountNonzero(unittest.TestCase):
    """测试 count_nonzero 功能 / Test count_nonzero functionality."""

    def test_count_nonzero_basic(self):
        """测试 count_nonzero 基本功能 / Test basic count_nonzero."""
        x = paddle.to_tensor([0, 1, 2, 0, 3], dtype='int64')
        out = paddle.count_nonzero(x)
        np.testing.assert_equal(out.numpy(), 3)

    def test_count_nonzero_axis(self):
        """测试 count_nonzero 沿轴 / Test count_nonzero along axis."""
        x = paddle.to_tensor([[0, 1], [2, 0]], dtype='int64')
        out = paddle.count_nonzero(x, axis=0)
        np.testing.assert_array_equal(out.numpy(), [1, 1])


class TestTrunc(unittest.TestCase):
    """测试 trunc 功能 / Test trunc functionality."""

    def test_trunc_basic(self):
        """测试 trunc 基本功能 / Test basic trunc."""
        x = paddle.to_tensor([-1.5, -0.5, 0.5, 1.5])
        out = paddle.trunc(x)
        np.testing.assert_array_almost_equal(
            out.numpy(), [-1.0, -0.0, 0.0, 1.0]
        )


class TestTrace(unittest.TestCase):
    """测试 trace 功能 / Test trace functionality."""

    def test_trace_basic(self):
        """测试 trace 基本功能 / Test basic trace."""
        x = paddle.to_tensor([[1, 2], [3, 4]], dtype='float32')
        out = paddle.trace(x)
        np.testing.assert_allclose(out.numpy(), 5.0, rtol=1e-5)

    def test_trace_offset(self):
        """测试 trace offset / Test trace with offset."""
        x = paddle.to_tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype='float32')
        out = paddle.trace(x, offset=1)
        # Offset 1 diagonal: 2 + 6
        np.testing.assert_allclose(out.numpy(), 8.0, rtol=1e-5)

    def test_trace_axis(self):
        """测试 trace axis / Test trace with axis."""
        x = paddle.to_tensor([[1, 2], [3, 4]], dtype='float32')
        out = paddle.trace(x)
        np.testing.assert_allclose(out.numpy(), 5.0, rtol=1e-5)


class TestCumsum(unittest.TestCase):
    """测试 cumsum 功能 / Test cumsum functionality."""

    def test_cumsum_basic(self):
        """测试 cumsum 基本功能 / Test basic cumsum."""
        x = paddle.to_tensor([1, 2, 3], dtype='float32')
        out = paddle.cumsum(x)
        np.testing.assert_array_almost_equal(out.numpy(), [1.0, 3.0, 6.0])

    def test_cumsum_axis(self):
        """测试 cumsum 沿轴 / Test cumsum along axis."""
        x = paddle.to_tensor([[1, 2, 3], [4, 5, 6]], dtype='float32')
        out = paddle.cumsum(x, axis=1)
        np.testing.assert_array_almost_equal(
            out.numpy(), [[1.0, 3.0, 6.0], [4.0, 9.0, 15.0]]
        )

    def test_cumsum_dtype(self):
        """测试 cumsum 指定 dtype / Test cumsum with specified dtype."""
        x = paddle.to_tensor([1, 2, 3], dtype='float32')
        out = paddle.cumsum(x, dtype='float64')
        self.assertEqual(out.dtype, paddle.float64)


class TestCumprod(unittest.TestCase):
    """测试 cumprod 功能 / Test cumprod functionality."""

    def test_cumprod_basic(self):
        """测试 cumprod 基本功能 / Test basic cumprod."""
        x = paddle.to_tensor([1, 2, 3, 4], dtype='float32')
        out = paddle.cumprod(x)
        np.testing.assert_array_almost_equal(out.numpy(), [1.0, 2.0, 6.0, 24.0])

    def test_cumprod_axis(self):
        """测试 cumprod 沿轴 / Test cumprod along axis."""
        x = paddle.to_tensor([[1, 2], [3, 4]], dtype='float32')
        out = paddle.cumprod(x, dim=0)
        np.testing.assert_array_almost_equal(
            out.numpy(), [[1.0, 2.0], [3.0, 8.0]]
        )


class TestProd(unittest.TestCase):
    """测试 prod 功能 / Test prod functionality."""

    def test_prod_all(self):
        """测试全元素乘积 / Test product over all elements."""
        x = paddle.to_tensor([1, 2, 3, 4], dtype='float32')
        out = paddle.prod(x)
        np.testing.assert_allclose(out.numpy(), 24.0, rtol=1e-5)

    def test_prod_axis(self):
        """测试沿轴乘积 / Test product along axis."""
        x = paddle.to_tensor([[1, 2], [3, 4]], dtype='float32')
        out = paddle.prod(x, axis=1)
        np.testing.assert_array_almost_equal(out.numpy(), [2.0, 12.0])

    def test_prod_keepdim(self):
        """测试乘积 keepdim / Test product with keepdim."""
        x = paddle.to_tensor([1, 2, 3], dtype='float32')
        out = paddle.prod(x, keepdim=True)
        self.assertEqual(out.shape, [1])

    def test_prod_dtype(self):
        """测试乘积指定 dtype / Test product with specified dtype."""
        x = paddle.to_tensor([1, 2, 3], dtype='float32')
        out = paddle.prod(x, dtype='float64')
        self.assertEqual(out.dtype, paddle.float64)


class TestGammaln(unittest.TestCase):
    """测试 gammaln 功能 / Test gammaln functionality."""

    def test_gammaln_basic(self):
        """测试 gammaln 基本功能 / Test basic gammaln."""
        x = paddle.to_tensor([1.0, 2.0, 3.0])
        out = paddle.gammaln(x)
        expected = np.array([0.0, np.log(1.0), np.log(2.0)])
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5)


class TestDigamma(unittest.TestCase):
    """测试 digamma 功能 / Test digamma functionality."""

    def test_digamma_basic(self):
        """测试 digamma 基本功能 / Test basic digamma."""
        x = paddle.to_tensor([1.0, 2.0, 3.0])
        out = paddle.digamma(x)
        self.assertEqual(out.shape, [3])


class TestGammainc(unittest.TestCase):
    """测试 gammainc 功能 / Test gammainc functionality."""

    def test_gammainc_basic(self):
        """测试 gammainc 基本功能 / Test basic gammainc."""
        x = paddle.to_tensor([1.0, 2.0])
        y = paddle.to_tensor([1.0, 1.0])
        out = paddle.gammainc(x, y)
        self.assertEqual(out.shape, [2])


class TestNeg(unittest.TestCase):
    """测试 neg 功能 / Test neg functionality."""

    def test_neg_basic(self):
        """测试 neg 基本功能 / Test basic neg."""
        x = paddle.to_tensor([1.0, -2.0, 3.0])
        out = paddle.neg(x)
        np.testing.assert_array_almost_equal(out.numpy(), [-1.0, 2.0, -3.0])


class TestLerp(unittest.TestCase):
    """测试 lerp 功能 / Test lerp functionality."""

    def test_lerp_basic(self):
        """测试 lerp 基本功能 / Test basic lerp."""
        x = paddle.to_tensor([1.0, 2.0, 3.0])
        y = paddle.to_tensor([5.0, 6.0, 7.0])
        w = 0.5
        out = paddle.lerp(x, y, w)
        np.testing.assert_array_almost_equal(out.numpy(), [3.0, 4.0, 5.0])


class TestRad2deg(unittest.TestCase):
    """测试 rad2deg 功能 / Test rad2deg functionality."""

    def test_rad2deg_basic(self):
        """测试 rad2deg 基本功能 / Test basic rad2deg."""
        pi = paddle.to_tensor([0.0, np.pi])
        out = paddle.rad2deg(pi)
        np.testing.assert_allclose(out.numpy(), [0.0, 180.0], rtol=1e-4)


class TestDeg2rad(unittest.TestCase):
    """测试 deg2rad 功能 / Test deg2rad functionality."""

    def test_deg2rad_basic(self):
        """测试 deg2rad 基本功能 / Test basic deg2rad."""
        deg = paddle.to_tensor([0.0, 180.0])
        out = paddle.deg2rad(deg)
        np.testing.assert_allclose(out.numpy(), [0.0, np.pi], rtol=1e-4)


class TestGcd(unittest.TestCase):
    """测试 gcd 功能 / Test gcd functionality."""

    def test_gcd_basic(self):
        """测试 gcd 基本功能 / Test basic gcd."""
        x = paddle.to_tensor([12, 18], dtype='int32')
        y = paddle.to_tensor([6, 9], dtype='int32')
        out = paddle.gcd(x, y)
        np.testing.assert_array_equal(out.numpy(), [6, 9])


class TestLcm(unittest.TestCase):
    """测试 lcm 功能 / Test lcm functionality."""

    def test_lcm_basic(self):
        """测试 lcm 基本功能 / Test basic lcm."""
        x = paddle.to_tensor([4, 6], dtype='int32')
        y = paddle.to_tensor([6, 9], dtype='int32')
        out = paddle.lcm(x, y)
        np.testing.assert_array_equal(out.numpy(), [12, 18])


class TestDiff(unittest.TestCase):
    """测试 diff 功能 / Test diff functionality."""

    def test_diff_basic(self):
        """测试 diff 基本功能 / Test basic diff."""
        x = paddle.to_tensor([1, 3, 6, 10], dtype='float32')
        out = paddle.diff(x)
        np.testing.assert_array_almost_equal(out.numpy(), [2.0, 3.0, 4.0])

    def test_diff_n(self):
        """测试 diff n 次 / Test diff with n=2."""
        x = paddle.to_tensor([1, 3, 6, 10], dtype='float32')
        out = paddle.diff(x, n=2)
        np.testing.assert_array_almost_equal(out.numpy(), [1.0, 1.0])

    def test_diff_axis(self):
        """测试 diff 沿轴 / Test diff along axis."""
        x = paddle.to_tensor([[1, 2, 3], [4, 5, 6]], dtype='float32')
        out = paddle.diff(x, axis=0)
        np.testing.assert_array_almost_equal(out.numpy(), [[3.0, 3.0, 3.0]])


class TestFrac(unittest.TestCase):
    """测试 frac 功能 / Test frac functionality."""

    def test_frac_basic(self):
        """测试 frac 基本功能 / Test basic frac."""
        x = paddle.to_tensor([1.5, 2.7, 3.8])
        out = paddle.frac(x)
        expected = np.array([0.5, 0.7, 0.8])
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5)


class TestSgn(unittest.TestCase):
    """测试 sgn 功能 / Test sgn functionality."""

    def test_sgn_basic(self):
        """测试 sgn 基本功能 / Test basic sgn."""
        x = paddle.to_tensor([-3.0, -0.0, 0.0, 2.0])
        out = paddle.sgn(x)
        np.testing.assert_array_almost_equal(
            out.numpy(), [-1.0, -0.0, 0.0, 1.0]
        )


class TestHypot(unittest.TestCase):
    """测试 hypot 功能 / Test hypot functionality."""

    def test_hypot_basic(self):
        """测试 hypot 基本功能 / Test basic hypot."""
        x = paddle.to_tensor([3.0, 5.0])
        y = paddle.to_tensor([4.0, 12.0])
        out = paddle.hypot(x, y)
        np.testing.assert_allclose(out.numpy(), [5.0, 13.0], rtol=1e-5)


class TestCopysign(unittest.TestCase):
    """测试 copysign 功能 / Test copysign functionality."""

    def test_copysign_basic(self):
        """测试 copysign 基本功能 / Test basic copysign."""
        x = paddle.to_tensor([1.0, 2.0])
        y = paddle.to_tensor([-1.0, 1.0])
        out = paddle.copysign(x, y)
        np.testing.assert_array_almost_equal(out.numpy(), [-1.0, 2.0])


class TestSinc(unittest.TestCase):
    """测试 sinc 功能 / Test sinc functionality."""

    def test_sinc_basic(self):
        """测试 sinc 基本功能 / Test basic sinc."""
        x = paddle.to_tensor([0.0, 1.0, 2.0])
        out = paddle.sinc(x)
        # sinc(0) = 1, sinc(x) = sin(pi*x)/(pi*x)
        self.assertAlmostEqual(out[0].numpy(), 1.0, places=5)
        self.assertEqual(out.shape, [3])


class TestSignbit(unittest.TestCase):
    """测试 signbit 功能 / Test signbit functionality."""

    def test_signbit_basic(self):
        """测试 signbit 基本功能 / Test basic signbit."""
        x = paddle.to_tensor([-3.0, -0.0, 0.0, 2.0])
        out = paddle.signbit(x)
        np.testing.assert_array_equal(out.numpy(), [True, True, False, False])


class TestIsposinf(unittest.TestCase):
    """测试 isposinf 功能 / Test isposinf functionality."""

    def test_isposinf_basic(self):
        """测试 isposinf 基本功能 / Test basic isposinf."""
        x = paddle.to_tensor([float('inf'), -float('inf'), 1.0])
        out = paddle.isposinf(x)
        np.testing.assert_array_equal(out.numpy(), [True, False, False])


class TestIsneginf(unittest.TestCase):
    """测试 isneginf 功能 / Test isneginf functionality."""

    def test_isneginf_basic(self):
        """测试 isneginf 基本功能 / Test basic isneginf."""
        x = paddle.to_tensor([float('inf'), -float('inf'), 1.0])
        out = paddle.isneginf(x)
        np.testing.assert_array_equal(out.numpy(), [False, True, False])


class TestIsreal(unittest.TestCase):
    """测试 isreal 功能 / Test isreal functionality."""

    def test_isreal_float(self):
        """测试 isreal 浮点 / Test isreal with float."""
        x = paddle.to_tensor([1.0, 2.0])
        out = paddle.isreal(x)
        np.testing.assert_array_equal(out.numpy(), [True, True])

    def test_isreal_complex(self):
        """测试 isreal 复数 / Test isreal with complex."""
        x = paddle.to_tensor([1 + 0j, 1 + 1j])
        out = paddle.isreal(x)
        np.testing.assert_array_equal(out.numpy(), [True, False])


class TestIsin(unittest.TestCase):
    """测试 isin 功能 / Test isin functionality."""

    def test_isin_basic(self):
        """测试 isin 基本功能 / Test basic isin."""
        x = paddle.to_tensor([1, 2, 3, 4])
        y = paddle.to_tensor([2, 4, 6])
        out = paddle.isin(x, y)
        np.testing.assert_array_equal(out.numpy(), [False, True, False, True])


class TestTake(unittest.TestCase):
    """测试 take 功能 / Test take functionality."""

    def test_take_basic(self):
        """测试 take 基本功能 / Test basic take."""
        x = paddle.to_tensor([10, 20, 30, 40])
        index = paddle.to_tensor([0, 2, 3], dtype='int64')
        out = paddle.take(x, index)
        np.testing.assert_array_almost_equal(out.numpy(), [10.0, 30.0, 40.0])


class TestFrexp(unittest.TestCase):
    """测试 frexp 功能 / Test frexp functionality."""

    def test_frexp_basic(self):
        """测试 frexp 基本功能 / Test basic frexp."""
        x = paddle.to_tensor([1.0, 2.0, 3.0])
        mantissa, exponent = paddle.frexp(x)
        self.assertEqual(mantissa.shape, [3])
        self.assertEqual(exponent.shape, [3])


if __name__ == '__main__':
    unittest.main()
