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

# [AUTO-GENERATED] Test file for paddle.amp
# 覆盖模块: paddle/amp/
# Uncovered lines: Various amp module functions

import unittest

import paddle


class TestAMPAutoCast(unittest.TestCase):
    """测试 AMP 自动混合精度
    Test AMP auto mixed precision"""

    def test_autocast_context(self):
        """测试 autocast 上下文管理器
        Test autocast context manager"""
        x = paddle.randn([4, 4], dtype='float32')
        with paddle.amp.auto_cast():
            y = paddle.matmul(x, x)
        self.assertIsNotNone(y)

    def test_autocast_list(self):
        """测试 autocast 白名单
        Test autocast with custom list"""
        x = paddle.randn([4, 4], dtype='float32')
        with paddle.amp.auto_cast(custom_white_list=['matmul']):
            y = paddle.matmul(x, x)
        self.assertIsNotNone(y)

    @unittest.skipIf(not paddle.is_compiled_with_cuda(), "Requires CUDA")
    def test_autocast_gpu(self):
        """测试 GPU 上的 autocast
        Test autocast on GPU"""
        x = paddle.randn([4, 4], dtype='float32')
        x = x.cuda()
        with paddle.amp.auto_cast():
            y = paddle.matmul(x, x)
        self.assertIsNotNone(y)


class TestAMPGradScaler(unittest.TestCase):
    """测试 AMP GradScaler
    Test AMP GradScaler"""

    def test_grad_scaler_init(self):
        """测试 GradScaler 初始化
        Test GradScaler initialization"""
        scaler = paddle.amp.GradScaler(init_loss_scaling=1024.0)
        self.assertIsNotNone(scaler)

    def test_grad_scaler_default(self):
        """测试 GradScaler 默认参数
        Test GradScaler with default parameters"""
        scaler = paddle.amp.GradScaler()
        self.assertIsNotNone(scaler)

    def test_grad_scaler_step(self):
        """测试 GradScaler step
        Test GradScaler step"""
        linear = paddle.nn.Linear(10, 10)
        scaler = paddle.amp.GradScaler(init_loss_scaling=1024.0)
        optimizer = paddle.optimizer.Adam(
            parameters=linear.parameters(), learning_rate=0.001
        )
        x = paddle.randn([4, 10])
        with paddle.amp.auto_cast():
            y = linear(x)
            loss = y.mean()
        scaled_loss = scaler.scale(loss)
        scaled_loss.backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.clear_grad()


if __name__ == '__main__':
    unittest.main()
