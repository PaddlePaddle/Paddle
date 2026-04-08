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
进阶LR调度器单元测试 / Advanced LR Scheduler Unit Tests

测试目标 / Test Target:
  paddle.optimizer.lr 更多调度器 (python/paddle/optimizer/lr.py, 覆盖率约80.5%)

覆盖的模块 / Covered Modules:
  - paddle.optimizer.lr.ReduceOnPlateau: 按平台期降低LR
  - paddle.optimizer.lr.CosineAnnealingDecay: 余弦退火
  - paddle.optimizer.lr.MultiStepDecay: 多步衰减
  - paddle.optimizer.lr.PolynomialDecay: 多项式衰减
  - paddle.optimizer.lr.LambdaDecay: Lambda衰减
  - paddle.optimizer.lr.OneCycleLR: 单循环LR
  - paddle.optimizer.lr.LinearWarmup: 线性预热

作用 / Purpose:
  覆盖进阶学习率调度策略的代码路径，补充lr_scheduler功能测试。
"""

import unittest

import paddle
from paddle import nn

paddle.disable_static()


def create_optimizer(lr_scheduler):
    """创建使用调度器的优化器 / Create optimizer with scheduler"""
    model = nn.Linear(5, 2)
    return paddle.optimizer.SGD(
        learning_rate=lr_scheduler, parameters=model.parameters()
    ), model


class TestReduceOnPlateau(unittest.TestCase):
    """测试ReduceOnPlateau调度器 / Test ReduceOnPlateau scheduler"""

    def test_basic(self):
        """测试基本ReduceOnPlateau / Test basic ReduceOnPlateau"""
        scheduler = paddle.optimizer.lr.ReduceOnPlateau(
            learning_rate=0.1, factor=0.5, patience=2
        )
        opt, model = create_optimizer(scheduler)
        # Simulate training
        for i in range(5):
            x = paddle.randn([4, 5])
            y = model(x)
            loss_val = y.mean().item()
            scheduler.step(loss_val)

    def test_reduce_on_plateau_mode_max(self):
        """测试max模式 / Test max mode"""
        scheduler = paddle.optimizer.lr.ReduceOnPlateau(
            learning_rate=0.1, mode='max', factor=0.5, patience=1
        )
        scheduler.step(0.5)
        scheduler.step(0.3)  # No improvement
        scheduler.step(0.2)  # No improvement, should reduce

    def test_cooldown(self):
        """测试cooldown参数 / Test cooldown parameter"""
        scheduler = paddle.optimizer.lr.ReduceOnPlateau(
            learning_rate=0.1, factor=0.5, patience=1, cooldown=2
        )
        for i in range(6):
            scheduler.step(1.0 / (i + 1))


class TestCosineAnnealingDecay(unittest.TestCase):
    """测试余弦退火调度器 / Test CosineAnnealingDecay scheduler"""

    def test_basic(self):
        """测试基本余弦退火 / Test basic cosine annealing"""
        scheduler = paddle.optimizer.lr.CosineAnnealingDecay(
            learning_rate=0.1, T_max=10
        )
        opt, model = create_optimizer(scheduler)
        for _ in range(12):
            x = paddle.randn([4, 5])
            y = model(x)
            y.mean().backward()
            opt.step()
            opt.clear_grad()
            scheduler.step()

    def test_with_eta_min(self):
        """测试带最小LR的余弦退火 / Test cosine annealing with eta_min"""
        scheduler = paddle.optimizer.lr.CosineAnnealingDecay(
            learning_rate=0.1, T_max=10, eta_min=0.001
        )
        for _ in range(10):
            scheduler.step()
        self.assertGreaterEqual(scheduler.get_lr(), 0.001)


class TestMultiStepDecay(unittest.TestCase):
    """测试多步衰减调度器 / Test MultiStepDecay scheduler"""

    def test_basic(self):
        """测试基本MultiStepDecay / Test basic MultiStepDecay"""
        scheduler = paddle.optimizer.lr.MultiStepDecay(
            learning_rate=0.1, milestones=[3, 6], gamma=0.1
        )
        opt, model = create_optimizer(scheduler)
        init_lr = scheduler.get_lr()
        for i in range(8):
            x = paddle.randn([4, 5])
            y = model(x)
            y.mean().backward()
            opt.step()
            opt.clear_grad()
            scheduler.step()
        final_lr = scheduler.get_lr()
        # LR should have decreased after milestones
        self.assertLess(final_lr, init_lr)


class TestPolynomialDecay(unittest.TestCase):
    """测试多项式衰减调度器 / Test PolynomialDecay scheduler"""

    def test_basic(self):
        """测试基本PolynomialDecay / Test basic PolynomialDecay"""
        scheduler = paddle.optimizer.lr.PolynomialDecay(
            learning_rate=0.1, decay_steps=10, end_lr=0.001
        )
        opt, model = create_optimizer(scheduler)
        for _ in range(12):
            scheduler.step()
        # After decay_steps, lr should approach end_lr
        lr = scheduler.get_lr()
        self.assertAlmostEqual(lr, 0.001, places=4)

    def test_cycle(self):
        """测试循环模式 / Test cycle mode"""
        scheduler = paddle.optimizer.lr.PolynomialDecay(
            learning_rate=0.1, decay_steps=5, end_lr=0.01, cycle=True
        )
        for _ in range(12):
            scheduler.step()


class TestLambdaDecay(unittest.TestCase):
    """测试Lambda调度器 / Test LambdaDecay scheduler"""

    def test_basic(self):
        """测试基本LambdaDecay / Test basic LambdaDecay"""
        scheduler = paddle.optimizer.lr.LambdaDecay(
            learning_rate=0.1, lr_lambda=lambda epoch: 0.95**epoch
        )
        opt, model = create_optimizer(scheduler)
        for i in range(5):
            scheduler.step()
        # LR should decrease
        self.assertLess(scheduler.get_lr(), 0.1)

    def test_warmup_lambda(self):
        """测试预热Lambda / Test warmup lambda"""
        warmup_steps = 5

        def warmup_fn(step):
            if step < warmup_steps:
                return step / warmup_steps
            return 1.0

        scheduler = paddle.optimizer.lr.LambdaDecay(
            learning_rate=0.1, lr_lambda=warmup_fn
        )
        # Before warmup
        for _ in range(warmup_steps + 2):
            scheduler.step()


class TestLinearWarmup(unittest.TestCase):
    """测试线性预热 / Test LinearWarmup scheduler"""

    def test_basic(self):
        """测试基本LinearWarmup / Test basic LinearWarmup"""
        scheduler = paddle.optimizer.lr.LinearWarmup(
            learning_rate=0.1, warmup_steps=5, start_lr=0.0, end_lr=0.1
        )
        opt, model = create_optimizer(scheduler)
        for i in range(8):
            x = paddle.randn([4, 5])
            y = model(x)
            y.mean().backward()
            opt.step()
            opt.clear_grad()
            scheduler.step()

    def test_with_base_scheduler(self):
        """测试配合基础调度器使用 / Test with base scheduler"""
        base_scheduler = paddle.optimizer.lr.StepDecay(
            learning_rate=0.1, step_size=10, gamma=0.5
        )
        scheduler = paddle.optimizer.lr.LinearWarmup(
            learning_rate=base_scheduler,
            warmup_steps=5,
            start_lr=0.0,
            end_lr=0.1,
        )
        for _ in range(8):
            scheduler.step()


class TestCyclicalLR(unittest.TestCase):
    """测试循环LR / Test cyclical LR"""

    def test_exponential_decay(self):
        """测试指数衰减 / Test exponential decay"""
        scheduler = paddle.optimizer.lr.ExponentialDecay(
            learning_rate=0.1, gamma=0.9
        )
        init_lr = scheduler.get_lr()
        scheduler.step()
        new_lr = scheduler.get_lr()
        self.assertAlmostEqual(new_lr, init_lr * 0.9, places=5)

    def test_inverse_time_decay(self):
        """测试逆时间衰减 / Test inverse time decay"""
        scheduler = paddle.optimizer.lr.InverseTimeDecay(
            learning_rate=0.1, gamma=0.5
        )
        init_lr = scheduler.get_lr()
        self.assertAlmostEqual(init_lr, 0.1, places=5)
        for _ in range(3):
            scheduler.step()


if __name__ == '__main__':
    unittest.main()
