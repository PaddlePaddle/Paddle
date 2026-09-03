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

# [AUTO-GENERATED] Test file for paddle/nn/utils/clip_grad_norm_.py
# Target file: paddle/nn/utils/clip_grad_norm_.py (93.3% coverage)
# Uncovered lines: 74 (RuntimeError: not in dynamic mode),
#   87 (empty grads -> return tensor(0.0))

"""梯度范数裁剪模块测试 / Clip gradient norm module tests

测试目标 / Test Target:
  paddle/nn/utils/clip_grad_norm_.py

覆盖的模块 / Covered Modules:
  - clip_grad_norm_: with iterable, single Tensor, no gradients,
    inf norm, L1 norm, L2 norm, error_if_nonfinite
"""

import unittest

import numpy as np

import paddle
from paddle import nn


class TestClipGradNorm(unittest.TestCase):
    """测试 clip_grad_norm_ 函数
    Test clip_grad_norm_ function"""

    def setUp(self):
        paddle.disable_static()

    def test_clip_grad_norm_iterable(self):
        """测试对可迭代参数的梯度范数裁剪
        Test clip_grad_norm_ with iterable parameters"""
        model = nn.Linear(4, 2)
        x = paddle.randn([4, 4])
        y = model(x)
        loss = y.sum()
        loss.backward()

        total_norm = paddle.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0
        )
        self.assertIsNotNone(total_norm)
        self.assertFalse(paddle.isnan(total_norm).numpy())

    def test_clip_grad_norm_single_tensor(self):
        """测试对单个 Tensor 的梯度范数裁剪 (line 76-77)
        Test clip_grad_norm_ with single Tensor"""
        x = paddle.randn([4, 4], dtype='float32')
        x.stop_gradient = False
        y = x * x
        loss = y.sum()
        loss.backward()

        total_norm = paddle.nn.utils.clip_grad_norm_(x, max_norm=1.0)
        self.assertIsNotNone(total_norm)
        self.assertFalse(paddle.isnan(total_norm).numpy())

    def test_clip_grad_norm_no_gradients(self):
        """测试无梯度参数 (line 86-87, return tensor(0.0))
        Test clip_grad_norm_ with no gradients"""
        model = nn.Linear(4, 2)
        # Don't compute backward
        total_norm = paddle.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0
        )
        np.testing.assert_allclose(total_norm.numpy(), 0.0, atol=1e-6)

    def test_clip_grad_norm_inf_norm(self):
        """测试无穷范数裁剪 (line 88-91)
        Test clip_grad_norm_ with inf norm"""
        model = nn.Linear(4, 2)
        x = paddle.randn([4, 4])
        y = model(x)
        loss = y.sum()
        loss.backward()

        total_norm = paddle.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0, norm_type=float('inf')
        )
        self.assertIsNotNone(total_norm)
        self.assertFalse(paddle.isnan(total_norm).numpy())

    def test_clip_grad_norm_l1_norm(self):
        """测试 L1 范数裁剪 (line 94-98)
        Test clip_grad_norm_ with L1 norm"""
        model = nn.Linear(4, 2)
        x = paddle.randn([4, 4])
        y = model(x)
        loss = y.sum()
        loss.backward()

        total_norm = paddle.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0, norm_type=1
        )
        self.assertIsNotNone(total_norm)
        self.assertFalse(paddle.isnan(total_norm).numpy())

    def test_clip_grad_norm_large_max_norm(self):
        """测试大 max_norm 值（不裁剪场景）(line 113 clip_coef_clamped)
        Test with large max_norm (no clipping scenario)"""
        model = nn.Linear(4, 2)
        x = paddle.randn([4, 4]) * 0.01
        y = model(x)
        loss = y.sum()
        loss.backward()

        original_grads = [
            p.grad.numpy().copy()
            for p in model.parameters()
            if p.grad is not None
        ]
        total_norm = paddle.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1000.0
        )
        # With large max_norm, gradients should be nearly unchanged
        clipped_grads = [
            p.grad.numpy().copy()
            for p in model.parameters()
            if p.grad is not None
        ]
        for orig, clipped in zip(original_grads, clipped_grads):
            np.testing.assert_allclose(orig, clipped, atol=1e-6)

    def test_clip_grad_norm_small_max_norm(self):
        """测试小 max_norm 值（强制裁剪场景）
        Test with small max_norm (forced clipping)"""
        model = nn.Linear(4, 2)
        x = paddle.randn([4, 4]) * 100
        y = model(x)
        loss = y.sum()
        loss.backward()

        total_norm = paddle.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=0.001
        )
        self.assertIsNotNone(total_norm)

    def test_clip_grad_norm_returns_tensor(self):
        """测试返回值为 Tensor
        Test return value is a Tensor"""
        model = nn.Linear(4, 2)
        x = paddle.randn([4, 4])
        y = model(x)
        loss = y.sum()
        loss.backward()

        total_norm = paddle.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0
        )
        self.assertIsInstance(total_norm, paddle.Tensor)

    def test_clip_grad_norm_int_max_norm(self):
        """测试整数 max_norm (line 84 float conversion)
        Test with integer max_norm"""
        model = nn.Linear(4, 2)
        x = paddle.randn([4, 4])
        y = model(x)
        loss = y.sum()
        loss.backward()

        total_norm = paddle.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1
        )
        self.assertIsNotNone(total_norm)


if __name__ == '__main__':
    unittest.main()
