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

# [AUTO-GENERATED] Test file for paddle.optimizer.asgd and paddle.optimizer.adam
# 覆盖模块: paddle/optimizer/asgd.py, paddle/optimizer/adam.py
# 未覆盖行: asgd: 146,161,196,208,222,224,228,232,314,316,321,325,338,340,349,355,357,358,359,361,369,371,375; adam: 277,278,280,330,424,441,445,449,458,459,528,529,548,553,570,654,656,694,695,696,697,698,699,703,706,707,708,712,715,716
# Covered module: paddle/optimizer/asgd.py, paddle/optimizer/adam.py
# Uncovered lines: asgd: 146,161,196,208,222,224,228,232,314,316,321,325,338,340,349,355,357-361,369,371,375; adam: 277,278,280,330,424,441,445,449,458,459,528,529,548,553,570,654,656,694-699,703,706-708,712,715,716

import unittest

import paddle


class TestASGD(unittest.TestCase):
    """测试 ASGD 优化器
    Test ASGD optimizer"""

    def test_asgd_basic(self):
        """测试基本的 ASGD 优化器
        Test basic ASGD optimizer"""
        linear = paddle.nn.Linear(10, 10)
        opt = paddle.optimizer.ASGD(
            parameters=linear.parameters(), learning_rate=0.01
        )
        x = paddle.randn([4, 10])
        y = linear(x)
        loss = y.mean()
        loss.backward()
        opt.step()
        opt.clear_grad()

    def test_asgd_with_weight_decay(self):
        """测试带 weight_decay 的 ASGD
        Test ASGD with weight_decay"""
        linear = paddle.nn.Linear(10, 10)
        opt = paddle.optimizer.ASGD(
            parameters=linear.parameters(),
            learning_rate=0.01,
            weight_decay=0.001,
        )
        x = paddle.randn([4, 10])
        y = linear(x)
        loss = y.mean()
        loss.backward()
        opt.step()
        opt.clear_grad()

    def test_asgd_with_grad_clip(self):
        """测试带梯度裁剪的 ASGD
        Test ASGD with gradient clipping"""
        linear = paddle.nn.Linear(10, 10)
        clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0)
        opt = paddle.optimizer.ASGD(
            parameters=linear.parameters(),
            learning_rate=0.01,
            grad_clip=clip,
        )
        x = paddle.randn([4, 10])
        y = linear(x)
        loss = y.mean()
        loss.backward()
        opt.step()
        opt.clear_grad()


class TestAdam(unittest.TestCase):
    """测试 Adam 优化器
    Test Adam optimizer"""

    def test_adam_basic(self):
        """测试基本的 Adam 优化器
        Test basic Adam optimizer"""
        linear = paddle.nn.Linear(10, 10)
        opt = paddle.optimizer.Adam(
            parameters=linear.parameters(), learning_rate=0.001
        )
        x = paddle.randn([4, 10])
        y = linear(x)
        loss = y.mean()
        loss.backward()
        opt.step()
        opt.clear_grad()

    def test_adam_with_beta(self):
        """测试带自定义 beta 的 Adam
        Test Adam with custom beta values"""
        linear = paddle.nn.Linear(10, 10)
        opt = paddle.optimizer.Adam(
            parameters=linear.parameters(),
            learning_rate=0.001,
            beta1=0.9,
            beta2=0.999,
        )
        x = paddle.randn([4, 10])
        y = linear(x)
        loss = y.mean()
        loss.backward()
        opt.step()
        opt.clear_grad()

    def test_adam_with_weight_decay(self):
        """测试带 weight_decay 的 Adam
        Test Adam with weight_decay"""
        linear = paddle.nn.Linear(10, 10)
        opt = paddle.optimizer.Adam(
            parameters=linear.parameters(),
            learning_rate=0.001,
            weight_decay=0.01,
        )
        x = paddle.randn([4, 10])
        y = linear(x)
        loss = y.mean()
        loss.backward()
        opt.step()
        opt.clear_grad()

    def test_adam_with_grad_clip(self):
        """测试带梯度裁剪的 Adam
        Test Adam with gradient clipping"""
        linear = paddle.nn.Linear(10, 10)
        clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0)
        opt = paddle.optimizer.Adam(
            parameters=linear.parameters(),
            learning_rate=0.001,
            grad_clip=clip,
        )
        x = paddle.randn([4, 10])
        y = linear(x)
        loss = y.mean()
        loss.backward()
        opt.step()
        opt.clear_grad()

    def test_adam_with_amsgrad(self):
        """测试带 amsgrad 的 Adam
        Test Adam with amsgrad"""
        linear = paddle.nn.Linear(10, 10)
        opt = paddle.optimizer.Adam(
            parameters=linear.parameters(),
            learning_rate=0.001,
            amsgrad=True,
        )
        x = paddle.randn([4, 10])
        y = linear(x)
        loss = y.mean()
        loss.backward()
        opt.step()
        opt.clear_grad()


class TestAdamW(unittest.TestCase):
    """测试 AdamW 优化器
    Test AdamW optimizer"""

    def test_adamw_basic(self):
        """测试基本的 AdamW 优化器
        Test basic AdamW optimizer"""
        linear = paddle.nn.Linear(10, 10)
        opt = paddle.optimizer.AdamW(
            parameters=linear.parameters(), learning_rate=0.001
        )
        x = paddle.randn([4, 10])
        y = linear(x)
        loss = y.mean()
        loss.backward()
        opt.step()
        opt.clear_grad()

    def test_adamw_with_decay(self):
        """测试带 weight_decay 的 AdamW
        Test AdamW with weight_decay"""
        linear = paddle.nn.Linear(10, 10)
        opt = paddle.optimizer.AdamW(
            parameters=linear.parameters(),
            learning_rate=0.001,
            weight_decay=0.01,
        )
        x = paddle.randn([4, 10])
        y = linear(x)
        loss = y.mean()
        loss.backward()
        opt.step()
        opt.clear_grad()


if __name__ == '__main__':
    unittest.main()
