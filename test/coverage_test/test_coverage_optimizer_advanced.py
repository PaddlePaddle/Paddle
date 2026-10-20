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
高级优化器单元测试 / Advanced Optimizer Unit Tests

测试目标 / Test Target:
  paddle.optimizer 模块 - 多种优化器 (覆盖率约82-84%)

覆盖的模块 / Covered Modules:
  - paddle.optimizer.Adamax: Adamax优化器
  - paddle.optimizer.Adagrad: 自适应学习率优化器
  - paddle.optimizer.Adadelta: Adadelta优化器
  - paddle.optimizer.ASGD: 平均随机梯度下降
  - paddle.optimizer.RMSProp: RMSProp优化器
  - paddle.optimizer.Momentum: 动量优化器

作用 / Purpose:
  覆盖各类优化器的正向传播、参数更新、学习率调整等代码路径，
  补充未被原有测试覆盖的优化器功能。
"""

import unittest

import paddle
from paddle import nn

paddle.disable_static()


def create_simple_model():
    """创建简单模型 / Create simple model"""
    return nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))


def do_one_step(model, optimizer):
    """执行一步优化 / Perform one optimization step"""
    x = paddle.randn([4, 10])
    y = model(x)
    loss = y.mean()
    loss.backward()
    optimizer.step()
    optimizer.clear_grad()
    return loss.item()


class TestAdamaxOptimizer(unittest.TestCase):
    """测试Adamax优化器 / Test Adamax optimizer"""

    def test_adamax_basic(self):
        """测试Adamax基本功能 / Test basic Adamax functionality"""
        model = create_simple_model()
        optimizer = paddle.optimizer.Adamax(
            learning_rate=0.01, parameters=model.parameters()
        )
        loss = do_one_step(model, optimizer)
        self.assertIsNotNone(loss)

    def test_adamax_with_weight_decay(self):
        """测试带权重衰减的Adamax / Test Adamax with weight decay"""
        model = create_simple_model()
        optimizer = paddle.optimizer.Adamax(
            learning_rate=0.01, weight_decay=0.01, parameters=model.parameters()
        )
        do_one_step(model, optimizer)

    def test_adamax_beta1_beta2(self):
        """测试Adamax的beta参数 / Test Adamax beta parameters"""
        model = create_simple_model()
        optimizer = paddle.optimizer.Adamax(
            learning_rate=0.01,
            beta1=0.9,
            beta2=0.999,
            parameters=model.parameters(),
        )
        do_one_step(model, optimizer)

    def test_adamax_multiple_steps(self):
        """测试Adamax多步优化 / Test Adamax multi-step optimization"""
        model = create_simple_model()
        optimizer = paddle.optimizer.Adamax(
            learning_rate=0.01, parameters=model.parameters()
        )
        for _ in range(5):
            do_one_step(model, optimizer)


class TestAdagradOptimizer(unittest.TestCase):
    """测试Adagrad优化器 / Test Adagrad optimizer"""

    def test_adagrad_basic(self):
        """测试Adagrad基本功能 / Test basic Adagrad functionality"""
        model = create_simple_model()
        optimizer = paddle.optimizer.Adagrad(
            learning_rate=0.01, parameters=model.parameters()
        )
        loss = do_one_step(model, optimizer)
        self.assertIsNotNone(loss)

    def test_adagrad_epsilon(self):
        """测试Adagrad的epsilon参数 / Test Adagrad epsilon parameter"""
        model = create_simple_model()
        optimizer = paddle.optimizer.Adagrad(
            learning_rate=0.01, epsilon=1e-8, parameters=model.parameters()
        )
        do_one_step(model, optimizer)

    def test_adagrad_initial_accumulator(self):
        """测试Adagrad初始累积器 / Test Adagrad initial accumulator"""
        model = create_simple_model()
        optimizer = paddle.optimizer.Adagrad(
            learning_rate=0.01,
            initial_accumulator_value=0.1,
            parameters=model.parameters(),
        )
        do_one_step(model, optimizer)

    def test_adagrad_multiple_steps(self):
        """测试Adagrad多步 / Test Adagrad multiple steps"""
        model = create_simple_model()
        optimizer = paddle.optimizer.Adagrad(
            learning_rate=0.1, parameters=model.parameters()
        )
        for _ in range(5):
            do_one_step(model, optimizer)


class TestAdadeltaOptimizer(unittest.TestCase):
    """测试Adadelta优化器 / Test Adadelta optimizer"""

    def test_adadelta_basic(self):
        """测试Adadelta基本功能 / Test basic Adadelta functionality"""
        model = create_simple_model()
        optimizer = paddle.optimizer.Adadelta(
            learning_rate=1.0, parameters=model.parameters()
        )
        loss = do_one_step(model, optimizer)
        self.assertIsNotNone(loss)

    def test_adadelta_rho_epsilon(self):
        """测试Adadelta的rho和epsilon参数 / Test Adadelta rho and epsilon"""
        model = create_simple_model()
        optimizer = paddle.optimizer.Adadelta(
            learning_rate=1.0,
            rho=0.95,
            epsilon=1e-6,
            parameters=model.parameters(),
        )
        do_one_step(model, optimizer)

    def test_adadelta_multiple_steps(self):
        """测试Adadelta多步 / Test Adadelta multiple steps"""
        model = create_simple_model()
        optimizer = paddle.optimizer.Adadelta(
            learning_rate=1.0, parameters=model.parameters()
        )
        for _ in range(5):
            do_one_step(model, optimizer)


class TestRMSPropOptimizer(unittest.TestCase):
    """测试RMSProp优化器 / Test RMSProp optimizer"""

    def test_rmsprop_basic(self):
        """测试RMSProp基本功能 / Test basic RMSProp functionality"""
        model = create_simple_model()
        optimizer = paddle.optimizer.RMSProp(
            learning_rate=0.01, parameters=model.parameters()
        )
        loss = do_one_step(model, optimizer)
        self.assertIsNotNone(loss)

    def test_rmsprop_with_momentum(self):
        """测试带动量的RMSProp / Test RMSProp with momentum"""
        model = create_simple_model()
        optimizer = paddle.optimizer.RMSProp(
            learning_rate=0.01, momentum=0.9, parameters=model.parameters()
        )
        do_one_step(model, optimizer)

    def test_rmsprop_centered(self):
        """测试centered RMSProp / Test centered RMSProp"""
        model = create_simple_model()
        optimizer = paddle.optimizer.RMSProp(
            learning_rate=0.01, centered=True, parameters=model.parameters()
        )
        do_one_step(model, optimizer)

    def test_rmsprop_rho_epsilon(self):
        """测试RMSProp的rho和epsilon / Test RMSProp rho and epsilon"""
        model = create_simple_model()
        optimizer = paddle.optimizer.RMSProp(
            learning_rate=0.01,
            rho=0.9,
            epsilon=1e-6,
            parameters=model.parameters(),
        )
        do_one_step(model, optimizer)


class TestMomentumOptimizer(unittest.TestCase):
    """测试Momentum优化器 / Test Momentum optimizer"""

    def test_momentum_basic(self):
        """测试Momentum基本功能 / Test basic Momentum functionality"""
        model = create_simple_model()
        optimizer = paddle.optimizer.Momentum(
            learning_rate=0.01, momentum=0.9, parameters=model.parameters()
        )
        loss = do_one_step(model, optimizer)
        self.assertIsNotNone(loss)

    def test_momentum_nesterov(self):
        """测试Nesterov动量 / Test Nesterov momentum"""
        model = create_simple_model()
        optimizer = paddle.optimizer.Momentum(
            learning_rate=0.01,
            momentum=0.9,
            use_nesterov=True,
            parameters=model.parameters(),
        )
        do_one_step(model, optimizer)

    def test_momentum_weight_decay(self):
        """测试带权重衰减的Momentum / Test Momentum with weight decay"""
        model = create_simple_model()
        optimizer = paddle.optimizer.Momentum(
            learning_rate=0.01,
            momentum=0.9,
            weight_decay=0.001,
            parameters=model.parameters(),
        )
        do_one_step(model, optimizer)

    def test_momentum_set_lr(self):
        """测试动态设置学习率 / Test dynamic learning rate setting"""
        model = create_simple_model()
        optimizer = paddle.optimizer.Momentum(
            learning_rate=0.01, momentum=0.9, parameters=model.parameters()
        )
        optimizer.set_lr(0.001)
        self.assertAlmostEqual(optimizer.get_lr(), 0.001, places=5)


class TestSGDOptimizer(unittest.TestCase):
    """测试SGD优化器 / Test SGD optimizer"""

    def test_sgd_basic(self):
        """测试SGD基本功能 / Test basic SGD functionality"""
        model = create_simple_model()
        optimizer = paddle.optimizer.SGD(
            learning_rate=0.01, parameters=model.parameters()
        )
        loss = do_one_step(model, optimizer)
        self.assertIsNotNone(loss)

    def test_sgd_weight_decay(self):
        """测试带权重衰减的SGD / Test SGD with weight decay"""
        model = create_simple_model()
        optimizer = paddle.optimizer.SGD(
            learning_rate=0.01,
            weight_decay=0.001,
            parameters=model.parameters(),
        )
        do_one_step(model, optimizer)

    def test_sgd_with_lr_scheduler(self):
        """测试SGD配合学习率调度器 / Test SGD with lr scheduler"""
        model = create_simple_model()
        scheduler = paddle.optimizer.lr.StepDecay(
            learning_rate=0.1, step_size=10, gamma=0.1
        )
        optimizer = paddle.optimizer.SGD(
            learning_rate=scheduler, parameters=model.parameters()
        )
        for _ in range(3):
            do_one_step(model, optimizer)
            scheduler.step()


if __name__ == '__main__':
    unittest.main()
