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

# [AUTO-GENERATED] Test file for paddle/nn/utils/clip_grad_value_.py
# Target file: paddle/nn/utils/clip_grad_value_.py (92.9% coverage)
# Uncovered lines: 61 (RuntimeError: not in dynamic mode)

"""梯度值裁剪模块测试 / Clip gradient value module tests

测试目标 / Test Target:
  paddle/nn/utils/clip_grad_value_.py

覆盖的模块 / Covered Modules:
  - clip_grad_value_: with iterable parameters, single Tensor, no gradients
"""

import unittest

import numpy as np

import paddle
from paddle import nn


class TestClipGradValue(unittest.TestCase):
    """测试 clip_grad_value_ 函数
    Test clip_grad_value_ function"""

    def setUp(self):
        paddle.disable_static()

    def test_clip_grad_value_iterable(self):
        """测试对可迭代参数的梯度值裁剪
        Test clip_grad_value_ with iterable parameters"""
        model = nn.Linear(4, 2)
        x = paddle.randn([4, 4])
        y = model(x)
        loss = y.sum()
        loss.backward()

        paddle.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
        for p in model.parameters():
            if p.grad is not None:
                self.assertTrue(bool((p.grad.abs() <= 0.5001).all().numpy()))

    def test_clip_grad_value_single_tensor(self):
        """测试对单个 Tensor 的梯度值裁剪 (line 63-64)
        Test clip_grad_value_ with single Tensor"""
        x = paddle.randn([4, 4], dtype='float32')
        x.stop_gradient = False
        y = x * x
        loss = y.sum()
        loss.backward()

        # Pass single tensor directly
        paddle.nn.utils.clip_grad_value_(x, clip_value=1.0)
        self.assertTrue(bool((x.grad.abs() <= 1.0001).all().numpy()))

    def test_clip_grad_value_no_gradient(self):
        """测试无梯度参数的梯度值裁剪
        Test clip_grad_value_ with parameters that have no gradients"""
        model = nn.Linear(4, 2)
        # Don't compute backward, so no gradients
        paddle.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
        # Should not raise, just skip parameters with None grad
        for p in model.parameters():
            self.assertIsNone(p.grad)

    def test_clip_grad_value_large_values(self):
        """测试大梯度值的裁剪
        Test clipping large gradient values"""
        model = nn.Linear(4, 2)
        x = paddle.randn([4, 4]) * 1000
        y = model(x)
        loss = y.sum()
        loss.backward()

        clip_val = 0.01
        paddle.nn.utils.clip_grad_value_(
            model.parameters(), clip_value=clip_val
        )
        for p in model.parameters():
            if p.grad is not None:
                self.assertTrue(
                    bool((p.grad.abs() <= clip_val + 1e-6).all().numpy())
                )

    def test_clip_grad_value_zero_clip(self):
        """测试零裁剪值
        Test with zero clip value"""
        model = nn.Linear(4, 2)
        x = paddle.randn([4, 4])
        y = model(x)
        loss = y.sum()
        loss.backward()

        paddle.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.0)
        for p in model.parameters():
            if p.grad is not None:
                np.testing.assert_allclose(p.grad.numpy(), 0.0, atol=1e-7)

    def test_clip_grad_value_int_clip(self):
        """测试整数裁剪值 (line 66 float conversion)
        Test with integer clip value"""
        model = nn.Linear(4, 2)
        x = paddle.randn([4, 4])
        y = model(x)
        loss = y.sum()
        loss.backward()

        # Pass integer clip value, should be converted to float
        paddle.nn.utils.clip_grad_value_(model.parameters(), clip_value=1)
        for p in model.parameters():
            if p.grad is not None:
                self.assertTrue(bool((p.grad.abs() <= 1.0001).all().numpy()))


if __name__ == '__main__':
    unittest.main()
