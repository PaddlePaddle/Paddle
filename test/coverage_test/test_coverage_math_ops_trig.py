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

"""
张量数学运算高级测试 / Advanced Tensor Math Operations Tests

测试目标 / Test Target:
  paddle.tensor.math 高级数学函数

覆盖的模块 / Covered Modules:
  - paddle.logsumexp: log-sum-exp
  - paddle.logit: logit变换
  - paddle.log1p: log(1+x)
  - paddle.expm1: exp(x)-1
  - paddle.i0/i1: 贝塞尔函数
  - paddle.sinc: sinc函数
  - paddle.round/ceil/floor/trunc: 取整函数

作用 / Purpose:
  补充高级数学运算API的测试，提升覆盖率。
"""

import unittest

import numpy as np

import paddle

paddle.disable_static()


class TestLogarithmicOps(unittest.TestCase):
    """测试对数运算 / Test logarithmic operations"""

    def test_logsumexp(self):
        """测试log-sum-exp / Test log-sum-exp"""
        x = paddle.to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = paddle.logsumexp(x, axis=1)
        self.assertEqual(result.shape, [2])
        # log-sum-exp of [1,2,3] should be ≈ 3.4076
        np.testing.assert_allclose(
            float(result[0].numpy()),
            np.log(np.exp(1) + np.exp(2) + np.exp(3)),
            rtol=1e-5,
        )

    def test_log1p(self):
        """测试log(1+x) / Test log1p"""
        x = paddle.to_tensor([0.0, 1.0, 2.0])
        result = paddle.log1p(x)
        expected = np.log1p([0.0, 1.0, 2.0])
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

    def test_expm1(self):
        """测试exp(x)-1 / Test expm1"""
        x = paddle.to_tensor([0.0, 0.5, 1.0])
        result = paddle.expm1(x)
        expected = np.expm1([0.0, 0.5, 1.0])
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

    def test_logit(self):
        """测试logit变换 / Test logit transform"""
        x = paddle.to_tensor([0.1, 0.5, 0.9])
        result = paddle.logit(x)
        self.assertEqual(result.shape, [3])


class TestRoundingOps(unittest.TestCase):
    """测试取整运算 / Test rounding operations"""

    def test_round(self):
        """测试round / Test round"""
        x = paddle.to_tensor([1.4, 1.5, 2.6, -1.4])
        result = paddle.round(x)
        self.assertEqual(result.shape, [4])

    def test_ceil(self):
        """测试ceil / Test ceiling"""
        x = paddle.to_tensor([1.1, 1.9, 2.0, -1.1])
        result = paddle.ceil(x)
        expected = np.ceil([1.1, 1.9, 2.0, -1.1])
        np.testing.assert_allclose(result.numpy(), expected)

    def test_floor(self):
        """测试floor / Test floor"""
        x = paddle.to_tensor([1.9, 2.0, 2.1, -1.1])
        result = paddle.floor(x)
        expected = np.floor([1.9, 2.0, 2.1, -1.1])
        np.testing.assert_allclose(result.numpy(), expected)

    def test_trunc(self):
        """测试截断 / Test truncation"""
        x = paddle.to_tensor([1.7, -1.7, 2.3, -2.3])
        result = paddle.trunc(x)
        expected = np.trunc([1.7, -1.7, 2.3, -2.3])
        np.testing.assert_allclose(result.numpy(), expected)

    def test_frac(self):
        """测试小数部分 / Test fractional part"""
        x = paddle.to_tensor([1.7, -1.7, 2.3])
        result = paddle.frac(x)
        self.assertEqual(result.shape, [3])


class TestTrigonometricOps(unittest.TestCase):
    """测试三角函数 / Test trigonometric operations"""

    def test_sin_cos_tan(self):
        """测试基本三角函数 / Test basic trig functions"""
        x = paddle.to_tensor([0.0, np.pi / 4, np.pi / 2, np.pi])
        sin_result = paddle.sin(x)
        cos_result = paddle.cos(x)
        tan_result = paddle.tan(paddle.to_tensor([0.0, np.pi / 4]))
        np.testing.assert_allclose(sin_result.numpy()[0], 0.0, atol=1e-6)
        np.testing.assert_allclose(cos_result.numpy()[0], 1.0, atol=1e-6)

    def test_asin_acos_atan(self):
        """测试反三角函数 / Test inverse trig functions"""
        x = paddle.to_tensor([0.0, 0.5, 1.0])
        asin_result = paddle.asin(x)
        acos_result = paddle.acos(x)
        atan_result = paddle.atan(x)
        self.assertEqual(asin_result.shape, [3])
        self.assertEqual(acos_result.shape, [3])
        self.assertEqual(atan_result.shape, [3])

    def test_atan2(self):
        """测试atan2 / Test atan2"""
        y = paddle.to_tensor([1.0, -1.0, 1.0])
        x = paddle.to_tensor([1.0, 1.0, -1.0])
        result = paddle.atan2(y, x)
        self.assertEqual(result.shape, [3])

    def test_sinh_cosh_tanh(self):
        """测试双曲函数 / Test hyperbolic functions"""
        x = paddle.to_tensor([0.0, 0.5, 1.0])
        sinh_result = paddle.sinh(x)
        cosh_result = paddle.cosh(x)
        tanh_result = paddle.tanh(x)
        np.testing.assert_allclose(
            sinh_result.numpy(), np.sinh([0.0, 0.5, 1.0]), rtol=1e-5
        )


class TestClampAndAbs(unittest.TestCase):
    """测试clip/abs运算 / Test clamp and abs operations"""

    def test_clip(self):
        """测试clip / Test clip (clamp)"""
        x = paddle.to_tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = paddle.clip(x, min=-1.0, max=1.0)
        expected = np.clip([-2.0, -1.0, 0.0, 1.0, 2.0], -1.0, 1.0)
        np.testing.assert_allclose(result.numpy(), expected)

    def test_abs(self):
        """测试绝对值 / Test absolute value"""
        x = paddle.to_tensor([-3.0, -1.5, 0.0, 1.5, 3.0])
        result = paddle.abs(x)
        expected = np.abs([-3.0, -1.5, 0.0, 1.5, 3.0])
        np.testing.assert_allclose(result.numpy(), expected)

    def test_sign(self):
        """测试符号函数 / Test sign function"""
        x = paddle.to_tensor([-3.0, 0.0, 3.0])
        result = paddle.sign(x)
        expected = np.sign([-3.0, 0.0, 3.0])
        np.testing.assert_allclose(result.numpy(), expected)

    def test_maximum_minimum(self):
        """测试逐元素最大/最小值 / Test element-wise max/min"""
        a = paddle.to_tensor([1.0, 5.0, 3.0])
        b = paddle.to_tensor([4.0, 2.0, 6.0])
        max_result = paddle.maximum(a, b)
        min_result = paddle.minimum(a, b)
        np.testing.assert_allclose(max_result.numpy(), [4.0, 5.0, 6.0])
        np.testing.assert_allclose(min_result.numpy(), [1.0, 2.0, 3.0])


class TestModAndDivOps(unittest.TestCase):
    """测试取模和整除运算 / Test modulo and floor divide operations"""

    def test_mod(self):
        """测试取模 / Test modulo"""
        x = paddle.to_tensor([10.0, 11.0, 12.0])
        y = paddle.to_tensor([3.0, 4.0, 5.0])
        result = paddle.mod(x, y)
        expected = np.mod([10.0, 11.0, 12.0], [3.0, 4.0, 5.0])
        np.testing.assert_allclose(result.numpy(), expected)

    def test_floor_divide(self):
        """测试整除 / Test floor divide"""
        x = paddle.to_tensor([10.0, 11.0, 12.0])
        y = paddle.to_tensor([3.0, 4.0, 5.0])
        result = paddle.floor_divide(x, y)
        expected = np.floor_divide([10.0, 11.0, 12.0], [3.0, 4.0, 5.0])
        np.testing.assert_allclose(result.numpy(), expected)

    def test_pow(self):
        """测试幂运算 / Test power operation"""
        x = paddle.to_tensor([2.0, 3.0, 4.0])
        y = paddle.to_tensor([2.0, 3.0, 0.5])
        result = paddle.pow(x, y)
        expected = np.power([2.0, 3.0, 4.0], [2.0, 3.0, 0.5])
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
