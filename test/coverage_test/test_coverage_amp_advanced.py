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
混合精度训练高级测试 / Advanced Mixed Precision Training Tests

测试目标 / Test Target:
  paddle AMP (Automatic Mixed Precision) 功能

覆盖的模块 / Covered Modules:
  - paddle.amp.auto_cast: 自动混合精度上下文
  - paddle.amp.GradScaler: 梯度缩放器
  - paddle.amp.decorate: AMP装饰器

作用 / Purpose:
  补充混合精度训练API的测试，提升覆盖率。
"""

import unittest

import numpy as np

import paddle
from paddle import nn

paddle.disable_static()


class TestAutocast(unittest.TestCase):
    """测试自动类型转换 / Test auto casting"""

    def test_autocast_basic(self):
        """测试基本autocast / Test basic autocast"""
        model = nn.Linear(4, 2)
        x = paddle.randn([4, 4])
        with paddle.amp.auto_cast():
            output = model(x)
        self.assertIsNotNone(output)

    def test_autocast_disable(self):
        """测试禁用autocast / Test disabled autocast"""
        model = nn.Linear(4, 2)
        x = paddle.randn([4, 4])
        with paddle.amp.auto_cast(enable=False):
            output = model(x)
        self.assertEqual(output.dtype, paddle.float32)

    def test_autocast_nested(self):
        """测试嵌套autocast / Test nested autocast"""
        model = nn.Linear(4, 2)
        x = paddle.randn([4, 4])
        with paddle.amp.auto_cast():
            y = model(x)
            with paddle.amp.auto_cast(enable=False):
                z = model(x)
        self.assertIsNotNone(y)
        self.assertIsNotNone(z)


class TestGradScaler(unittest.TestCase):
    """测试梯度缩放器 / Test gradient scaler"""

    def test_grad_scaler_basic(self):
        """测试基本梯度缩放 / Test basic gradient scaling"""
        model = nn.Linear(4, 2)
        optimizer = paddle.optimizer.Adam(parameters=model.parameters())
        scaler = paddle.amp.GradScaler(init_loss_scaling=1024)

        x = paddle.randn([4, 4])
        y = paddle.randn([4, 2])

        with paddle.amp.auto_cast():
            output = model(x)
            loss = nn.functional.mse_loss(output, y)

        scaled_loss = scaler.scale(loss)
        scaled_loss.backward()
        scaler.step(optimizer)
        scaler.update()

    def test_grad_scaler_state(self):
        """测试梯度缩放器状态 / Test grad scaler state"""
        scaler = paddle.amp.GradScaler(init_loss_scaling=512)
        state = scaler.state_dict()
        self.assertIn('scale', state)

    def test_grad_scaler_save_load(self):
        """测试梯度缩放器保存加载 / Test grad scaler save/load"""
        scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
        state = scaler.state_dict()

        new_scaler = paddle.amp.GradScaler(init_loss_scaling=512)
        new_scaler.load_state_dict(state)
        new_state = new_scaler.state_dict()
        self.assertEqual(
            float(np.asarray(state['scale']).item()),
            float(np.asarray(new_state['scale']).item()),
        )


class TestAMPDecorate(unittest.TestCase):
    """测试AMP装饰器 / Test AMP decorate"""

    def test_decorate_model(self):
        """测试模型AMP装饰 / Test model AMP decoration"""
        model = nn.Linear(4, 2)
        optimizer = paddle.optimizer.Adam(parameters=model.parameters())
        model, optimizer = paddle.amp.decorate(
            models=model, optimizers=optimizer, level='O1'
        )
        x = paddle.randn([4, 4])
        with paddle.amp.auto_cast():
            output = model(x)
        self.assertIsNotNone(output)

    def test_decorate_level_o1(self):
        """测试O1级别AMP / Test O1 level AMP"""
        model = nn.Sequential(
            nn.Conv2D(3, 8, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2D(1)
        )
        optimizer = paddle.optimizer.Adam(parameters=model.parameters())
        model, optimizer = paddle.amp.decorate(
            models=model, optimizers=optimizer, level='O1'
        )
        x = paddle.randn([2, 3, 16, 16])
        with paddle.amp.auto_cast():
            output = model(x)
        self.assertIsNotNone(output)


class TestMixedPrecisionTraining(unittest.TestCase):
    """测试混合精度训练 / Test mixed precision training"""

    def test_full_amp_training_step(self):
        """测试完整AMP训练步骤 / Test full AMP training step"""
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        optimizer = paddle.optimizer.Adam(parameters=model.parameters())
        scaler = paddle.amp.GradScaler()

        x = paddle.randn([8, 4])
        y = paddle.randn([8, 2])

        with paddle.amp.auto_cast():
            output = model(x)
            loss = nn.functional.mse_loss(output, y)

        scaled_loss = scaler.scale(loss)
        scaled_loss.backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.clear_grad()

        self.assertIsNotNone(loss)


if __name__ == '__main__':
    unittest.main()
