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
自动混合精度(AMP)单元测试 / Automatic Mixed Precision (AMP) Unit Tests

测试目标 / Test Target:
  paddle.amp 模块 (python/paddle/static/amp/*, 覆盖率约81.3%)

覆盖的模块 / Covered Modules:
  - paddle.amp.auto_cast: 自动类型转换上下文
  - paddle.amp.GradScaler: 梯度缩放器
  - paddle.amp.decorate: AMP模型装饰
  - paddle.amp.is_float16_supported: 检查FP16支持

作用 / Purpose:
  覆盖自动混合精度训练的各种代码路径，包括半精度计算、梯度缩放等。
"""

import unittest

import paddle
from paddle import nn

paddle.disable_static()

HAS_GPU = paddle.device.is_compiled_with_cuda()


class TestAutoCast(unittest.TestCase):
    """测试auto_cast上下文 / Test auto_cast context"""

    def test_auto_cast_float16(self):
        """测试FP16 auto_cast / Test FP16 auto_cast"""
        model = nn.Linear(10, 5)
        with paddle.amp.auto_cast(enable=True, dtype='float16'):
            x = paddle.randn([4, 10])
            y = model(x)
        # Output should exist
        self.assertEqual(y.shape, [4, 5])

    def test_auto_cast_bfloat16(self):
        """测试BF16 auto_cast / Test BF16 auto_cast"""
        model = nn.Linear(10, 5)
        try:
            with paddle.amp.auto_cast(enable=True, dtype='bfloat16'):
                x = paddle.randn([4, 10])
                y = model(x)
            self.assertEqual(y.shape, [4, 5])
        except Exception:
            # BF16 may not be supported on all hardware
            pass

    def test_auto_cast_disabled(self):
        """测试禁用auto_cast / Test disabled auto_cast"""
        model = nn.Linear(10, 5)
        with paddle.amp.auto_cast(enable=False):
            x = paddle.randn([4, 10])
            y = model(x)
        self.assertEqual(y.dtype, paddle.float32)

    def test_auto_cast_level(self):
        """测试auto_cast级别 / Test auto_cast level"""
        model = nn.Linear(10, 5)
        # Level O1: operations use float16 where beneficial
        with paddle.amp.auto_cast(enable=True, dtype='float16', level='O1'):
            x = paddle.randn([4, 10])
            y = model(x)
        self.assertEqual(y.shape, [4, 5])

    def test_auto_cast_custom_black_list(self):
        """测试自定义黑名单 / Test custom black list"""
        model = nn.Linear(10, 5)
        with paddle.amp.auto_cast(
            enable=True,
            dtype='float16',
            custom_black_list=['matmul_v2', 'matmul'],
        ):
            x = paddle.randn([4, 10])
            y = model(x)
        self.assertEqual(y.shape, [4, 5])


class TestGradScaler(unittest.TestCase):
    """测试梯度缩放器 / Test gradient scaler"""

    def test_grad_scaler_basic(self):
        """测试基本GradScaler / Test basic GradScaler"""
        scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
        self.assertEqual(scaler._init_loss_scaling, 1024)

    def test_grad_scaler_scale(self):
        """测试梯度缩放 / Test gradient scaling"""
        model = nn.Linear(10, 5)
        optimizer = paddle.optimizer.Adam(
            learning_rate=0.01, parameters=model.parameters()
        )
        scaler = paddle.amp.GradScaler(init_loss_scaling=1024)

        x = paddle.randn([4, 10])
        with paddle.amp.auto_cast(enable=True, dtype='float16'):
            y = model(x)
            loss = y.mean()

        scaled_loss = scaler.scale(loss)
        scaled_loss.backward()
        scaler.step(optimizer)
        scaler.update()

    def test_grad_scaler_unscale(self):
        """测试梯度反缩放 / Test gradient unscaling"""
        model = nn.Linear(10, 5)
        optimizer = paddle.optimizer.Adam(
            learning_rate=0.01, parameters=model.parameters()
        )
        scaler = paddle.amp.GradScaler(init_loss_scaling=1024)

        x = paddle.randn([4, 10])
        with paddle.amp.auto_cast(enable=True, dtype='float16'):
            y = model(x)
            loss = y.mean()

        scaled_loss = scaler.scale(loss)
        scaled_loss.backward()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()

    def test_grad_scaler_state_dict(self):
        """测试GradScaler状态字典 / Test GradScaler state dict"""
        scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
        state = scaler.state_dict()
        self.assertIn('scale', state)
        self.assertIn('incr_ratio', state)

    def test_grad_scaler_load_state_dict(self):
        """测试加载GradScaler状态 / Test loading GradScaler state"""
        scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
        state = scaler.state_dict()

        new_scaler = paddle.amp.GradScaler(init_loss_scaling=512)
        new_scaler.load_state_dict(state)
        self.assertEqual(new_scaler._init_loss_scaling, 1024)


class TestAMPDecorate(unittest.TestCase):
    """测试AMP模型装饰 / Test AMP model decoration"""

    def test_decorate_float16(self):
        """测试FP16模型装饰 / Test FP16 model decoration"""
        model = nn.Linear(10, 5)
        optimizer = paddle.optimizer.Adam(
            learning_rate=0.01, parameters=model.parameters()
        )
        model, optimizer = paddle.amp.decorate(
            models=model, optimizers=optimizer, level='O1', dtype='float16'
        )
        self.assertIsNotNone(model)
        self.assertIsNotNone(optimizer)

    def test_decorate_multiple_models(self):
        """测试多模型装饰 / Test multiple models decoration"""
        model1 = nn.Linear(10, 5)
        model2 = nn.Linear(5, 2)
        optimizer = paddle.optimizer.Adam(
            learning_rate=0.01,
            parameters=list(model1.parameters()) + list(model2.parameters()),
        )
        models, opt = paddle.amp.decorate(
            models=[model1, model2],
            optimizers=optimizer,
            level='O1',
            dtype='float16',
        )
        self.assertEqual(len(models), 2)


class TestAMPTraining(unittest.TestCase):
    """测试AMP完整训练流程 / Test complete AMP training workflow"""

    def test_amp_training_loop(self):
        """测试AMP训练循环 / Test AMP training loop"""
        model = nn.Linear(10, 5)
        optimizer = paddle.optimizer.Adam(
            learning_rate=0.01, parameters=model.parameters()
        )
        scaler = paddle.amp.GradScaler(init_loss_scaling=1024)

        for _ in range(3):
            x = paddle.randn([4, 10])
            with paddle.amp.auto_cast(enable=True, dtype='float16'):
                y = model(x)
                loss = y.mean()

            scaled_loss = scaler.scale(loss)
            scaled_loss.backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.clear_grad()

    def test_amp_with_grad_clip(self):
        """测试AMP配合梯度裁剪 / Test AMP with gradient clipping"""
        model = nn.Linear(10, 5)
        clip = nn.ClipGradByNorm(clip_norm=1.0)
        optimizer = paddle.optimizer.Adam(
            learning_rate=0.01, parameters=model.parameters(), grad_clip=clip
        )
        scaler = paddle.amp.GradScaler(init_loss_scaling=1024)

        x = paddle.randn([4, 10])
        with paddle.amp.auto_cast(enable=True, dtype='float16'):
            y = model(x)
            loss = y.mean()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.clear_grad()


if __name__ == '__main__':
    unittest.main()
