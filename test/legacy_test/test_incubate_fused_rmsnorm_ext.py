#  Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np

import paddle
from paddle.incubate.nn.functional import fused_rms_norm_ext

# 假设 fused_rms_norm_ext 已经被导入
# from your_module import fused_rms_norm_ext


class TestFusedRMSNorm(unittest.TestCase):
    def setUp(self):
        # 设置随机种子以确保结果可复现
        paddle.seed(2023)
        np.random.seed(2023)

    def rms_norm_reference(self, x, scale, bias=None, epsilon=1e-5):
        """
        使用 Paddle 原生操作实现 RMS Normalization 作为参考
        """
        # 计算均方根
        variance = paddle.mean(paddle.square(x), axis=-1, keepdim=True)
        # 计算 RMS
        rms = paddle.sqrt(variance + epsilon)
        # 归一化
        y = x / rms
        # 应用缩放
        y = y * scale.reshape([1, -1])
        # 应用偏置（如果有）
        if bias is not None:
            y = y + bias.reshape([1, -1])

        # 返回归一化后的张量、均值（RMS Norm 中为0）和逆标准差
        return y, (1.0 / rms).squeeze(-1)

    def test_2d_input(self):
        # 测试 2D 输入
        rows, cols = 32, 64
        x = paddle.randn([rows, cols])
        scale = paddle.randn([cols])

        # 使用我们的实现
        y_fused, invvar_fused = fused_rms_norm_ext(x, scale)

        # 使用参考实现
        y_ref, invvar_ref = self.rms_norm_reference(x, scale)

        # 验证结果
        np.testing.assert_allclose(y_fused, y_ref, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(
            invvar_fused, invvar_ref, rtol=1e-5, atol=1e-5
        )

    def test_without_bias(self):
        # 测试没有偏置的情况
        rows, cols = 32, 64
        x = paddle.randn([rows, cols])
        scale = paddle.randn([cols])

        # 使用我们的实现
        y_fused, invvar_fused = fused_rms_norm_ext(x, scale)

        # 使用参考实现
        y_ref, invvar_ref = self.rms_norm_reference(x, scale)

        # 验证结果
        np.testing.assert_allclose(y_fused, y_ref, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(
            invvar_fused, invvar_ref, rtol=1e-5, atol=1e-5
        )

    def test_backward(self):
        # 测试反向传播
        rows, cols = 16, 32
        x = paddle.randn([rows, cols], dtype='float32')
        x.stop_gradient = False
        scale = paddle.randn([cols], dtype='float32')
        scale.stop_gradient = False

        # 前向传播
        y_fused, invvar = fused_rms_norm_ext(x, scale)

        # 计算损失并反向传播
        loss = paddle.mean(y_fused)
        loss.backward()

        # 获取梯度
        x_grad_fused = x.grad.clone()
        scale_grad_fused = scale.grad.clone()

        # 重置梯度
        x.clear_gradient()
        scale.clear_gradient()

        # 使用参考实现
        y_ref, invvar_ref = self.rms_norm_reference(x, scale)
        loss_ref = paddle.mean(y_ref)
        loss_ref.backward()

        # 获取参考梯度
        x_grad_ref = x.grad
        scale_grad_ref = scale.grad

        # 验证梯度
        np.testing.assert_allclose(
            x_grad_fused, x_grad_ref, rtol=1e-4, atol=1e-4
        )
        np.testing.assert_allclose(
            scale_grad_fused, scale_grad_ref, rtol=1e-4, atol=1e-4
        )


if __name__ == '__main__':
    unittest.main()
