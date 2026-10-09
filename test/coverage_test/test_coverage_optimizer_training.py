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
优化器进阶测试 / Advanced Optimizer Tests

测试目标 / Test Target:
  paddle.optimizer 各种优化器

覆盖的模块 / Covered Modules:
  - paddle.optimizer.Adam: Adam优化器
  - paddle.optimizer.SGD: 随机梯度下降
  - paddle.optimizer.Momentum: 动量优化器
  - 优化器状态字典
  - 梯度裁剪

作用 / Purpose:
  补充优化器API的高级测试，提升覆盖率。
"""

import unittest

import paddle
import paddle.optimizer as optim
from paddle import nn

paddle.disable_static()


class SimpleModel(nn.Layer):
    """简单测试模型 / Simple test model"""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)


def training_step(model, optimizer, x, y):
    """执行单步训练 / Execute single training step"""
    pred = model(x)
    loss = paddle.nn.functional.mse_loss(pred, y)
    loss.backward()
    optimizer.step()
    optimizer.clear_grad()
    return float(loss.numpy())


class TestAdamOptimizer(unittest.TestCase):
    """测试Adam优化器 / Test Adam optimizer"""

    def test_adam_basic(self):
        """测试基本Adam / Test basic Adam"""
        model = SimpleModel()
        optimizer = optim.Adam(
            parameters=model.parameters(), learning_rate=0.001
        )
        x = paddle.randn([8, 4])
        y = paddle.randn([8, 2])
        loss = training_step(model, optimizer, x, y)
        self.assertIsNotNone(loss)

    def test_adam_weight_decay(self):
        """测试带weight_decay的Adam / Test Adam with weight decay"""
        model = SimpleModel()
        optimizer = optim.Adam(
            parameters=model.parameters(),
            learning_rate=0.001,
            weight_decay=1e-4,
        )
        x = paddle.randn([8, 4])
        y = paddle.randn([8, 2])
        loss = training_step(model, optimizer, x, y)
        self.assertIsNotNone(loss)

    def test_adam_beta(self):
        """测试自定义beta的Adam / Test Adam with custom betas"""
        model = SimpleModel()
        optimizer = optim.Adam(
            parameters=model.parameters(),
            learning_rate=0.001,
            beta1=0.9,
            beta2=0.999,
        )
        x = paddle.randn([8, 4])
        y = paddle.randn([8, 2])
        for _ in range(3):
            training_step(model, optimizer, x, y)

    def test_adam_state_dict(self):
        """测试Adam状态字典 / Test Adam state dict"""
        model = SimpleModel()
        optimizer = optim.Adam(parameters=model.parameters())
        x = paddle.randn([4, 4])
        y = paddle.randn([4, 2])
        training_step(model, optimizer, x, y)
        state = optimizer.state_dict()
        self.assertIsNotNone(state)


class TestSGDOptimizer(unittest.TestCase):
    """测试SGD优化器 / Test SGD optimizer"""

    def test_sgd_basic(self):
        """测试基本SGD / Test basic SGD"""
        model = SimpleModel()
        optimizer = optim.SGD(parameters=model.parameters(), learning_rate=0.01)
        x = paddle.randn([8, 4])
        y = paddle.randn([8, 2])
        loss_before = float(paddle.nn.functional.mse_loss(model(x), y).numpy())
        for _ in range(10):
            training_step(model, optimizer, x, y)
        loss_after = float(paddle.nn.functional.mse_loss(model(x), y).numpy())
        # Loss should decrease after training
        self.assertLess(loss_after, loss_before)

    def test_sgd_momentum(self):
        """测试带动量的SGD / Test SGD with momentum"""
        model = SimpleModel()
        optimizer = optim.Momentum(
            parameters=model.parameters(), learning_rate=0.01, momentum=0.9
        )
        x = paddle.randn([8, 4])
        y = paddle.randn([8, 2])
        for _ in range(5):
            training_step(model, optimizer, x, y)


class TestGradientClipping(unittest.TestCase):
    """测试梯度裁剪 / Test gradient clipping"""

    def test_clip_by_norm(self):
        """测试按范数裁剪 / Test gradient clipping by norm"""
        model = SimpleModel()
        clip = paddle.nn.ClipGradByNorm(clip_norm=1.0)
        optimizer = optim.Adam(
            parameters=model.parameters(), learning_rate=0.001, grad_clip=clip
        )
        x = paddle.randn([8, 4])
        y = paddle.randn([8, 2])
        training_step(model, optimizer, x, y)

    def test_clip_by_global_norm(self):
        """测试按全局范数裁剪 / Test gradient clipping by global norm"""
        model = SimpleModel()
        clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0)
        optimizer = optim.Adam(
            parameters=model.parameters(), learning_rate=0.001, grad_clip=clip
        )
        x = paddle.randn([8, 4])
        y = paddle.randn([8, 2])
        training_step(model, optimizer, x, y)

    def test_clip_by_value(self):
        """测试按值裁剪 / Test gradient clipping by value"""
        model = SimpleModel()
        clip = paddle.nn.ClipGradByValue(min=-0.5, max=0.5)
        optimizer = optim.Adam(
            parameters=model.parameters(), learning_rate=0.001, grad_clip=clip
        )
        x = paddle.randn([8, 4])
        y = paddle.randn([8, 2])
        training_step(model, optimizer, x, y)


class TestOptimizerParameterGroups(unittest.TestCase):
    """测试优化器参数组 / Test optimizer parameter groups"""

    def test_different_lr_per_group(self):
        """测试不同学习率的参数组 / Test parameter groups with different LRs"""
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        # Different learning rates for different layers
        params = [
            {'params': model[0].parameters(), 'learning_rate': 0.01},
            {'params': model[2].parameters(), 'learning_rate': 0.001},
        ]
        optimizer = optim.Adam(parameters=params, learning_rate=0.005)
        x = paddle.randn([4, 4])
        y = paddle.randn([4, 2])
        pred = model(x)
        loss = paddle.nn.functional.mse_loss(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()


if __name__ == '__main__':
    unittest.main()
