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

# [AUTO-GENERATED] Test file for paddle/nn/utils/weight_norm_hook.py
# Target file: paddle/nn/utils/weight_norm_hook.py (86.4% coverage)
# Uncovered lines: 39 (l2_norm len==1 axis=0), 45 (static graph path),
#   47-50 (static graph helper), 59 (norm_except_dim dim==-1),
#   71 (norm_except_dim dim==ndims-1), 71-72 (norm_except_dim dim==0),
#   92-94 (weight_norm dim==ndims-1), 93-94 (weight_norm dim!= -1/0/ndims-1),
#   117 (remove_weight_norm not found error)

"""权重归一化 Hook 模块测试 / Weight normalization hook tests

测试目标 / Test Target:
  paddle/nn/utils/weight_norm_hook.py

覆盖的模块 / Covered Modules:
  - l2_norm: 1-D input, dynamic mode
  - norm_except_dim: dim=-1, dim=0, dim=ndims-1, middle dim
  - _weight_norm: dim=-1, dim=0, dim=ndims-1, other dim
  - WeightNorm: apply, compute_weight, __call__, remove
  - weight_norm: public API
  - remove_weight_norm: public API, error case
"""

import unittest

import numpy as np

import paddle
from paddle import nn
from paddle.nn.utils.weight_norm_hook import (
    WeightNorm,
    _weight_norm,
    norm_except_dim,
    remove_weight_norm,
    weight_norm,
)


class TestL2Norm(unittest.TestCase):
    """测试 l2_norm 函数
    Test l2_norm function"""

    def setUp(self):
        paddle.disable_static()

    def test_l2_norm_1d(self):
        """测试 l2_norm 对 1-D 输入 (line 38-39, axis=0)
        Test l2_norm with 1-D input"""
        from paddle.nn.utils.weight_norm_hook import l2_norm

        x = paddle.to_tensor([3.0, 4.0], dtype='float32')
        result = l2_norm(x, axis=0)
        np.testing.assert_allclose(result.numpy(), [5.0], atol=1e-5)

    def test_l2_norm_2d(self):
        """测试 l2_norm 对 2-D 输入
        Test l2_norm with 2-D input"""
        from paddle.nn.utils.weight_norm_hook import l2_norm

        x = paddle.to_tensor([[3.0, 4.0], [6.0, 8.0]], dtype='float32')
        result = l2_norm(x, axis=1)
        expected = [5.0, 10.0]
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-5)


class TestNormExceptDim(unittest.TestCase):
    """测试 norm_except_dim 函数
    Test norm_except_dim function"""

    def setUp(self):
        paddle.disable_static()

    def test_norm_except_dim_neg_one(self):
        """测试 norm_except_dim dim=-1 (line 65-66)
        Test norm_except_dim with dim=-1"""
        x = paddle.randn([2, 3, 4], dtype='float32')
        result = norm_except_dim(x, dim=-1)
        # dim=-1 -> sqrt(sum(square(x)))
        expected = paddle.sqrt(paddle.sum(paddle.square(x)) + 1e-12)
        np.testing.assert_allclose(result.numpy(), expected.numpy(), atol=1e-5)

    def test_norm_except_dim_zero(self):
        """测试 norm_except_dim dim=0 (line 67-68)
        Test norm_except_dim with dim=0"""
        x = paddle.randn([4, 8], dtype='float32')
        result = norm_except_dim(x, dim=0)
        self.assertEqual(result.shape, [4])

    def test_norm_except_dim_last_dim(self):
        """测试 norm_except_dim dim=ndims-1 (line 70-71)
        Test norm_except_dim with dim=ndims-1"""
        x = paddle.randn([4, 8], dtype='float32')
        result = norm_except_dim(x, dim=1)
        self.assertEqual(result.shape, [8])

    def test_norm_except_dim_middle_dim(self):
        """测试 norm_except_dim 中间维度 (line 73-78, recursive)
        Test norm_except_dim with middle dimension"""
        x = paddle.randn([2, 3, 4], dtype='float32')
        result = norm_except_dim(x, dim=1)
        # dim=1 on 3D tensor: ndims-1=2, dim != -1, 0, or 2
        # Goes to transpose path: swaps dim 0 and dim 1
        # Then calls norm_except_dim with dim=0 on [3, 2, 4]
        # Returns shape [3]
        self.assertEqual(result.shape[0], 3)


class TestWeightNormFunc(unittest.TestCase):
    """测试 _weight_norm 内部函数
    Test _weight_norm internal function"""

    def setUp(self):
        paddle.disable_static()

    def test_weight_norm_dim_neg_one(self):
        """测试 _weight_norm dim=-1 (line 85-86)
        Test _weight_norm with dim=-1"""
        v = paddle.randn([4, 8], dtype='float32')
        g = paddle.randn([8], dtype='float32').abs() + 0.1
        result = _weight_norm(v, g, dim=-1)
        self.assertEqual(result.shape, v.shape)

    def test_weight_norm_dim_zero(self):
        """测试 _weight_norm dim=0 (line 87-88)
        Test _weight_norm with dim=0"""
        v = paddle.randn([4, 8], dtype='float32')
        g = paddle.randn([4], dtype='float32').abs() + 0.1
        result = _weight_norm(v, g, dim=0)
        self.assertEqual(result.shape, v.shape)

    def test_weight_norm_dim_last(self):
        """测试 _weight_norm dim=ndims-1 (line 91-92)
        Test _weight_norm with dim=ndims-1"""
        v = paddle.randn([4, 8], dtype='float32')
        g = paddle.randn([8], dtype='float32').abs() + 0.1
        result = _weight_norm(v, g, dim=1)
        self.assertEqual(result.shape, v.shape)

    def test_weight_norm_4d_dim1(self):
        """测试 _weight_norm 4D dim=1 (line 95-104, middle dim transpose path)
        Test _weight_norm 4D with dim=1"""
        v = paddle.randn([3, 4, 5, 6], dtype='float32')
        g = paddle.randn([4], dtype='float32').abs() + 0.1
        result = _weight_norm(v, g, dim=1)
        self.assertEqual(result.shape, v.shape)

    def test_weight_norm_4d_dim2(self):
        """测试 _weight_norm 4D dim=2
        Test _weight_norm 4D with dim=2"""
        v = paddle.randn([3, 4, 5, 6], dtype='float32')
        g = paddle.randn([5], dtype='float32').abs() + 0.1
        result = _weight_norm(v, g, dim=2)
        self.assertEqual(result.shape, v.shape)


class TestWeightNormApply(unittest.TestCase):
    """测试 WeightNorm.apply 方法
    Test WeightNorm.apply method"""

    def setUp(self):
        paddle.disable_static()

    def test_apply_linear(self):
        """测试对 Linear 层应用 weight norm
        Test apply weight norm on Linear"""
        linear = nn.Linear(4, 8)
        weight_norm(linear, name='weight', dim=0)
        self.assertTrue(hasattr(linear, 'weight_g'))
        self.assertTrue(hasattr(linear, 'weight_v'))

    def test_apply_conv2d(self):
        """测试对 Conv2D 层应用 weight norm
        Test apply weight norm on Conv2D"""
        conv = nn.Conv2D(3, 8, 3)
        weight_norm(conv, name='weight', dim=0)
        self.assertTrue(hasattr(conv, 'weight_g'))
        self.assertTrue(hasattr(conv, 'weight_v'))

    def test_apply_none_dim(self):
        """测试 dim=None (line 116-117)
        Test apply with dim=None"""
        linear = nn.Linear(4, 8)
        weight_norm(linear, name='weight', dim=None)
        self.assertTrue(hasattr(linear, 'weight_g'))

    def test_apply_negative_dim(self):
        """测试负数 dim (line 140)
        Test apply with negative dim"""
        conv = nn.Conv2D(3, 8, 3)
        weight_norm(conv, name='weight', dim=-1)
        self.assertTrue(hasattr(conv, 'weight_g'))

    def test_double_apply_raises(self):
        """测试重复应用 weight norm 抛出异常 (line 128-133)
        Test double apply raises RuntimeError"""
        linear = nn.Linear(4, 8)
        weight_norm(linear, name='weight', dim=0)
        with self.assertRaises(RuntimeError):
            weight_norm(linear, name='weight', dim=0)

    def test_invalid_dim_raises(self):
        """测试无效 dim 抛出异常 (line 140-141)
        Test invalid dim raises AssertionError"""
        conv = nn.Conv2D(3, 8, 3)
        with self.assertRaises(AssertionError):
            weight_norm(conv, name='weight', dim=10)


class TestWeightNormComputeAndRemove(unittest.TestCase):
    """测试 WeightNorm 计算和移除
    Test WeightNorm compute and remove"""

    def setUp(self):
        paddle.disable_static()

    def test_compute_weight(self):
        """测试 compute_weight (line 121-124)
        Test compute_weight"""

        linear = nn.Linear(4, 8)
        weight_norm(linear, name='weight', dim=0)
        # Find the WeightNorm hook from forward_pre_hooks
        hook = None
        for k, h in linear._forward_pre_hooks.items():
            if isinstance(h, WeightNorm):
                hook = h
                break
        self.assertIsNotNone(hook)
        w = hook.compute_weight(linear)
        self.assertEqual(w.shape, [4, 8])

    def test_call_hook(self):
        """测试 __call__ hook (line 174-175)
        Test __call__ hook"""
        linear = nn.Linear(4, 8)
        weight_norm(linear, name='weight', dim=0)
        x = paddle.randn([2, 4])
        y = linear(x)
        self.assertEqual(y.shape, [2, 8])

    def test_remove_weight_norm(self):
        """测试移除 weight norm
        Test remove weight norm"""
        linear = nn.Linear(4, 8)
        weight_norm(linear, name='weight', dim=0)
        remove_weight_norm(linear, name='weight')
        self.assertFalse(hasattr(linear, 'weight_g'))
        self.assertFalse(hasattr(linear, 'weight_v'))
        # Should have weight back
        self.assertTrue(hasattr(linear, 'weight'))

    def test_remove_weight_norm_not_found(self):
        """测试移除不存在的 weight norm (line 260)
        Test remove_weight_norm when not found raises ValueError"""
        linear = nn.Linear(4, 8)
        with self.assertRaises(ValueError):
            remove_weight_norm(linear, name='weight')


if __name__ == '__main__':
    unittest.main()
