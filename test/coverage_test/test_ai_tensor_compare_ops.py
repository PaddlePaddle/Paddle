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
张量比较运算测试 / Tensor Comparison Operations Tests

测试目标 / Test Target:
  paddle.tensor.logic 比较和逻辑运算

覆盖的模块 / Covered Modules:
  - paddle.equal/not_equal: 相等比较
  - paddle.greater_than/less_than/etc: 大小比较
  - paddle.logical_and/or/not/xor: 逻辑运算
  - paddle.where: 条件选择
  - paddle.is_nan/is_inf: 特殊值检测

作用 / Purpose:
  补充比较和逻辑操作API的测试，提升覆盖率。
"""

import unittest

import numpy as np

import paddle

paddle.disable_static()


class TestComparisonOps(unittest.TestCase):
    """测试比较运算 / Test comparison operations"""

    def test_equal(self):
        """测试相等比较 / Test equality comparison"""
        x = paddle.to_tensor([1.0, 2.0, 3.0])
        y = paddle.to_tensor([1.0, 5.0, 3.0])
        result = paddle.equal(x, y)
        np.testing.assert_array_equal(result.numpy(), [True, False, True])

    def test_not_equal(self):
        """测试不等比较 / Test inequality comparison"""
        x = paddle.to_tensor([1.0, 2.0, 3.0])
        y = paddle.to_tensor([1.0, 5.0, 3.0])
        result = paddle.not_equal(x, y)
        np.testing.assert_array_equal(result.numpy(), [False, True, False])

    def test_greater_than(self):
        """测试大于比较 / Test greater than comparison"""
        x = paddle.to_tensor([1.0, 5.0, 3.0])
        y = paddle.to_tensor([2.0, 4.0, 3.0])
        result = paddle.greater_than(x, y)
        np.testing.assert_array_equal(result.numpy(), [False, True, False])

    def test_greater_equal(self):
        """测试大于等于比较 / Test greater than or equal comparison"""
        x = paddle.to_tensor([1.0, 5.0, 3.0])
        y = paddle.to_tensor([2.0, 4.0, 3.0])
        result = paddle.greater_equal(x, y)
        np.testing.assert_array_equal(result.numpy(), [False, True, True])

    def test_less_than(self):
        """测试小于比较 / Test less than comparison"""
        x = paddle.to_tensor([1.0, 5.0, 3.0])
        y = paddle.to_tensor([2.0, 4.0, 3.0])
        result = paddle.less_than(x, y)
        np.testing.assert_array_equal(result.numpy(), [True, False, False])

    def test_less_equal(self):
        """测试小于等于比较 / Test less than or equal comparison"""
        x = paddle.to_tensor([1.0, 5.0, 3.0])
        y = paddle.to_tensor([2.0, 4.0, 3.0])
        result = paddle.less_equal(x, y)
        np.testing.assert_array_equal(result.numpy(), [True, False, True])

    def test_equal_all(self):
        """测试所有元素相等 / Test element-wise all equal"""
        x = paddle.to_tensor([1.0, 2.0, 3.0])
        y = paddle.to_tensor([1.0, 2.0, 3.0])
        result = paddle.equal_all(x, y)
        self.assertTrue(bool(result.numpy()))


class TestLogicalOps(unittest.TestCase):
    """测试逻辑运算 / Test logical operations"""

    def test_logical_and(self):
        """测试逻辑与 / Test logical AND"""
        x = paddle.to_tensor([True, True, False, False])
        y = paddle.to_tensor([True, False, True, False])
        result = paddle.logical_and(x, y)
        np.testing.assert_array_equal(
            result.numpy(), [True, False, False, False]
        )

    def test_logical_or(self):
        """测试逻辑或 / Test logical OR"""
        x = paddle.to_tensor([True, True, False, False])
        y = paddle.to_tensor([True, False, True, False])
        result = paddle.logical_or(x, y)
        np.testing.assert_array_equal(result.numpy(), [True, True, True, False])

    def test_logical_not(self):
        """测试逻辑非 / Test logical NOT"""
        x = paddle.to_tensor([True, False, True])
        result = paddle.logical_not(x)
        np.testing.assert_array_equal(result.numpy(), [False, True, False])

    def test_logical_xor(self):
        """测试逻辑异或 / Test logical XOR"""
        x = paddle.to_tensor([True, True, False, False])
        y = paddle.to_tensor([True, False, True, False])
        result = paddle.logical_xor(x, y)
        np.testing.assert_array_equal(
            result.numpy(), [False, True, True, False]
        )


class TestWhereAndNonzero(unittest.TestCase):
    """测试where和nonzero / Test where and nonzero"""

    def test_where_condition(self):
        """测试条件选择 / Test conditional selection"""
        cond = paddle.to_tensor([True, False, True, False])
        x = paddle.to_tensor([1.0, 2.0, 3.0, 4.0])
        y = paddle.to_tensor([10.0, 20.0, 30.0, 40.0])
        result = paddle.where(cond, x, y)
        np.testing.assert_allclose(result.numpy(), [1.0, 20.0, 3.0, 40.0])

    def test_where_broadcast(self):
        """测试广播条件选择 / Test where with broadcasting"""
        cond = paddle.to_tensor([[True, False], [False, True]])
        x = paddle.ones([2, 2])
        y = paddle.zeros([2, 2])
        result = paddle.where(cond, x, y)
        np.testing.assert_allclose(result.numpy(), [[1.0, 0.0], [0.0, 1.0]])

    def test_nonzero(self):
        """测试非零索引 / Test nonzero indices"""
        x = paddle.to_tensor([0.0, 1.0, 0.0, 2.0, 3.0])
        result = paddle.nonzero(x)
        np.testing.assert_allclose(result.numpy().flatten(), [1, 3, 4])


class TestSpecialValueCheck(unittest.TestCase):
    """测试特殊值检测 / Test special value detection"""

    def test_is_nan(self):
        """测试NaN检测 / Test NaN detection"""
        x = paddle.to_tensor([1.0, float('nan'), 3.0])
        result = paddle.isnan(x)
        np.testing.assert_array_equal(result.numpy(), [False, True, False])

    def test_is_inf(self):
        """测试Inf检测 / Test Inf detection"""
        x = paddle.to_tensor([1.0, float('inf'), -float('inf')])
        result = paddle.isinf(x)
        np.testing.assert_array_equal(result.numpy(), [False, True, True])

    def test_is_finite(self):
        """测试有限值检测 / Test finite value detection"""
        x = paddle.to_tensor([1.0, float('inf'), float('nan')])
        result = paddle.isfinite(x)
        np.testing.assert_array_equal(result.numpy(), [True, False, False])


if __name__ == '__main__':
    unittest.main()
