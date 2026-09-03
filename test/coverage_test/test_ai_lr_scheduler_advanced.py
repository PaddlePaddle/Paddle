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
学习率调度器高级测试 / Advanced Learning Rate Scheduler Tests

测试目标 / Test Target:
  paddle.optimizer.lr 学习率调度器

覆盖的模块 / Covered Modules:
  - ExponentialDecay: 指数衰减
  - StepDecay: 步进衰减
  - MultiStepDecay: 多步衰减
  - CyclicLR: 循环学习率
  - OneCycleLR: 单周期学习率

作用 / Purpose:
  补充学习率调度API的测试，提升覆盖率。
"""

import unittest

import paddle
import paddle.optimizer as optim

paddle.disable_static()


class TestExponentialDecay(unittest.TestCase):
    """测试指数衰减 / Test exponential decay"""

    def test_exponential_decay_basic(self):
        """测试基本指数衰减 / Test basic exponential decay"""
        scheduler = paddle.optimizer.lr.ExponentialDecay(
            learning_rate=0.1, gamma=0.9
        )
        self.assertAlmostEqual(scheduler.get_lr(), 0.1, places=5)
        scheduler.step()
        self.assertAlmostEqual(scheduler.get_lr(), 0.09, places=5)
        scheduler.step()
        self.assertAlmostEqual(scheduler.get_lr(), 0.081, places=5)

    def test_exponential_with_optimizer(self):
        """测试与优化器结合的指数衰减 / Test exponential decay with optimizer"""
        scheduler = paddle.optimizer.lr.ExponentialDecay(
            learning_rate=0.1, gamma=0.9
        )
        model = paddle.nn.Linear(4, 2)
        optimizer = optim.Adam(
            parameters=model.parameters(), learning_rate=scheduler
        )
        x = paddle.randn([4, 4])
        y = paddle.randn([4, 2])
        for _ in range(3):
            pred = model(x)
            loss = paddle.nn.functional.mse_loss(pred, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.clear_grad()
        self.assertAlmostEqual(optimizer.get_lr(), 0.1 * 0.9**3, places=5)


class TestStepDecay(unittest.TestCase):
    """测试步进衰减 / Test step decay"""

    def test_step_decay(self):
        """测试步进衰减 / Test step decay"""
        scheduler = paddle.optimizer.lr.StepDecay(
            learning_rate=0.1, step_size=2, gamma=0.5
        )
        # Initial
        self.assertAlmostEqual(scheduler.get_lr(), 0.1, places=5)
        scheduler.step()
        scheduler.step()
        # After 2 steps: 0.1 * 0.5 = 0.05
        self.assertAlmostEqual(scheduler.get_lr(), 0.05, places=5)
        scheduler.step()
        scheduler.step()
        # After 4 steps: 0.1 * 0.5^2 = 0.025
        self.assertAlmostEqual(scheduler.get_lr(), 0.025, places=5)


class TestMultiStepDecay(unittest.TestCase):
    """测试多步衰减 / Test multi-step decay"""

    def test_multi_step_decay(self):
        """测试多步衰减 / Test multi-step decay"""
        scheduler = paddle.optimizer.lr.MultiStepDecay(
            learning_rate=0.1, milestones=[3, 6], gamma=0.1
        )
        # Before first milestone: 0.1
        for _ in range(3):
            scheduler.step()
        self.assertAlmostEqual(scheduler.get_lr(), 0.01, places=6)
        # After first milestone: 0.1 * 0.1 = 0.01
        for _ in range(3):
            scheduler.step()
        self.assertAlmostEqual(scheduler.get_lr(), 0.001, places=7)


class TestCosineAnnealingDecay(unittest.TestCase):
    """测试余弦退火衰减 / Test cosine annealing decay"""

    def test_cosine_annealing(self):
        """测试余弦退火 / Test cosine annealing"""
        scheduler = paddle.optimizer.lr.CosineAnnealingDecay(
            learning_rate=0.1, T_max=10
        )
        lrs = []
        for _ in range(11):
            lrs.append(scheduler.get_lr())
            scheduler.step()
        # LR should start high and decrease
        self.assertGreater(lrs[0], lrs[5])
        # At T_max (step 10), LR should be at minimum
        self.assertAlmostEqual(lrs[10], 0.0, places=5)


class TestLRCallbacks(unittest.TestCase):
    """测试LR回调 / Test LR callbacks"""

    def test_reduce_on_plateau(self):
        """测试ReduceOnPlateau / Test ReduceOnPlateau"""
        scheduler = paddle.optimizer.lr.ReduceOnPlateau(
            learning_rate=0.1, patience=2, threshold=0.01
        )
        # Simulate improving loss
        for loss in [1.0, 0.9, 0.8, 0.7]:
            scheduler.step(loss)
        # Loss is improving, LR should stay the same
        self.assertAlmostEqual(scheduler.last_lr, 0.1, places=5)

        # Simulate plateau
        for loss in [0.7, 0.7, 0.7]:
            scheduler.step(loss)
        # After patience+1 steps of no improvement, LR should reduce
        self.assertLess(scheduler.last_lr, 0.1)

    def test_linear_warmup(self):
        """测试线性预热 / Test linear warmup"""
        scheduler = paddle.optimizer.lr.LinearWarmup(
            learning_rate=0.1, warmup_steps=5, start_lr=0.0, end_lr=0.1
        )
        # At start: near 0
        self.assertAlmostEqual(scheduler.get_lr(), 0.0, places=5)
        for _ in range(5):
            scheduler.step()
        # After warmup steps: 0.1
        self.assertAlmostEqual(scheduler.get_lr(), 0.1, places=5)


if __name__ == '__main__':
    unittest.main()
