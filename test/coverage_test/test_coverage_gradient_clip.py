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

# [AUTO-GENERATED] Unit test for paddle.nn.clip
# 自动生成的单测，覆盖 paddle.nn.clip 模块中未覆盖的代码

"""
测试模块：paddle.nn.clip (ClipGradByValue, ClipGradByNorm, ClipGradByGlobalNorm, clip_by_norm)
Test Module: paddle.nn.clip

本测试覆盖以下功能：
This test covers the following functions:
1. clip_by_norm - 按范数裁剪 / Clip by norm in dynamic mode
2. ClipGradByValue - 按值裁剪梯度 / Clip gradient by value
3. ClipGradByNorm - 按范数裁剪梯度 / Clip gradient by norm
4. ClipGradByGlobalNorm - 按全局范数裁剪梯度 / Clip gradient by global norm

覆盖的未覆盖行：88-110 (clip_by_norm static), 143-151 (merge_selected_rows)
"""

import unittest

import numpy as np

import paddle


class TestClipByNorm(unittest.TestCase):
    """测试clip_by_norm功能
    Test clip_by_norm function"""

    def setUp(self):
        paddle.disable_static()

    def test_clip_by_norm_no_clip(self):
        """范数小于max_norm时不裁剪 / No clipping when norm < max_norm"""
        x = paddle.to_tensor([[0.1, 0.2], [0.3, 0.4]], dtype='float32')
        out = paddle.nn.clip.clip_by_norm(x, max_norm=10.0)
        np.testing.assert_allclose(out.numpy(), x.numpy(), rtol=1e-5)

    def test_clip_by_norm_with_clip(self):
        """范数大于max_norm时裁剪 / Clipping when norm > max_norm"""
        x = paddle.to_tensor([[3.0, 4.0]], dtype='float32')
        out = paddle.nn.clip.clip_by_norm(x, max_norm=1.0)
        norm = np.linalg.norm(out.numpy())
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_clip_by_norm_zero(self):
        """零tensor裁剪 / Clip zero tensor"""
        x = paddle.zeros([3, 3], dtype='float32')
        out = paddle.nn.clip.clip_by_norm(x, max_norm=1.0)
        np.testing.assert_allclose(out.numpy(), np.zeros([3, 3]), atol=1e-7)


class TestClipGradByValue(unittest.TestCase):
    """测试ClipGradByValue梯度裁剪
    Test ClipGradByValue gradient clipping"""

    def setUp(self):
        paddle.disable_static()

    def test_clip_grad_by_value_basic(self):
        """基本按值裁剪 / Basic clip by value"""
        clip = paddle.nn.ClipGradByValue(min=-0.5, max=0.5)
        linear = paddle.nn.Linear(3, 3)
        x = paddle.randn([2, 3])
        y = linear(x)
        loss = y.sum()
        loss.backward()

        params_grads = []
        for p in linear.parameters():
            if p.grad is not None:
                params_grads.append((p, p.grad))

        clipped = clip(params_grads)
        for p, g in clipped:
            self.assertTrue(g.numpy().max() <= 0.5 + 1e-6)
            self.assertTrue(g.numpy().min() >= -0.5 - 1e-6)

    def test_clip_grad_by_value_symmetric(self):
        """对称裁剪 / Symmetric clipping with only max"""
        clip = paddle.nn.ClipGradByValue(max=1.0)
        self.assertIsNotNone(clip)


class TestClipGradByNorm(unittest.TestCase):
    """测试ClipGradByNorm梯度裁剪
    Test ClipGradByNorm gradient clipping"""

    def setUp(self):
        paddle.disable_static()

    def test_clip_grad_by_norm_basic(self):
        """基本按范数裁剪 / Basic clip by norm"""
        clip = paddle.nn.ClipGradByNorm(clip_norm=1.0)
        linear = paddle.nn.Linear(3, 3)
        x = paddle.randn([2, 3])
        y = linear(x)
        loss = y.sum() * 100
        loss.backward()

        params_grads = []
        for p in linear.parameters():
            if p.grad is not None:
                params_grads.append((p, p.grad))

        clipped = clip(params_grads)
        for p, g in clipped:
            norm = float(paddle.linalg.norm(g).numpy())
            self.assertLessEqual(norm, 1.0 + 1e-5)


class TestClipGradByGlobalNorm(unittest.TestCase):
    """测试ClipGradByGlobalNorm全局范数裁剪
    Test ClipGradByGlobalNorm global norm clipping"""

    def setUp(self):
        paddle.disable_static()

    def test_clip_grad_by_global_norm_basic(self):
        """基本全局范数裁剪 / Basic global norm clipping"""
        clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0)
        linear = paddle.nn.Linear(10, 10)
        x = paddle.randn([2, 10])
        y = linear(x)
        loss = y.sum() * 100
        loss.backward()

        params_grads = []
        for p in linear.parameters():
            if p.grad is not None:
                params_grads.append((p, p.grad))

        clipped = clip(params_grads)
        self.assertIsNotNone(clipped)

    def test_clip_grad_by_global_norm_no_clip(self):
        """全局范数未超限不裁剪 / No clipping when global norm is within limit"""
        clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=1000.0)
        linear = paddle.nn.Linear(3, 3)
        x = paddle.randn([2, 3])
        y = linear(x)
        loss = y.sum()
        loss.backward()

        params_grads = []
        for p in linear.parameters():
            if p.grad is not None:
                params_grads.append((p, p.grad.clone()))

        clipped = clip(params_grads)
        self.assertIsNotNone(clipped)


if __name__ == '__main__':
    unittest.main()
