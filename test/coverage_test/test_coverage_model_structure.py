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
模型剪枝和压缩测试 / Model Pruning and Compression Tests

测试目标 / Test Target:
  paddle.nn 模型结构操作

覆盖的模块 / Covered Modules:
  - nn.Layer.parameters(): 参数访问
  - nn.Layer.sublayers(): 子层访问
  - nn.Layer.named_parameters(): 命名参数
  - 模型迭代和修改

作用 / Purpose:
  补充模型结构操作API的测试，提升覆盖率。
"""

import unittest

import numpy as np

import paddle
from paddle import nn

paddle.disable_static()


class TestModelStructure(unittest.TestCase):
    """测试模型结构 / Test model structure"""

    def setUp(self):
        """设置测试模型 / Setup test model"""
        self.model = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 2),
        )

    def test_parameters_count(self):
        """测试参数数量 / Test parameter count"""
        params = list(self.model.parameters())
        # 3 Linear layers, each with weight and bias = 6 params
        self.assertEqual(len(params), 6)

    def test_named_parameters(self):
        """测试命名参数 / Test named parameters"""
        named_params = dict(self.model.named_parameters())
        self.assertIn('0.weight', named_params)
        self.assertIn('0.bias', named_params)

    def test_sublayers(self):
        """测试子层 / Test sublayers"""
        sublayers = self.model.sublayers()
        self.assertEqual(len(sublayers), 5)  # 3 Linear + 2 ReLU

    def test_named_sublayers(self):
        """测试命名子层 / Test named sublayers"""
        named_sublayers = dict(self.model.named_sublayers())
        self.assertIn('0', named_sublayers)
        self.assertIn('2', named_sublayers)

    def test_total_params_count(self):
        """测试总参数量 / Test total parameter count"""
        total = sum(p.numel() for p in self.model.parameters())
        # Linear(4,8): 4*8+8=40, Linear(8,4): 8*4+4=36, Linear(4,2): 4*2+2=10 = 86
        self.assertEqual(total, 86)


class TestModelModification(unittest.TestCase):
    """测试模型修改 / Test model modification"""

    def test_freeze_parameters(self):
        """测试冻结参数 / Test freezing parameters"""
        model = nn.Linear(4, 2)
        # Freeze all parameters
        for param in model.parameters():
            param.stop_gradient = True
        # Verify all frozen
        for param in model.parameters():
            self.assertTrue(param.stop_gradient)

    def test_selective_freeze(self):
        """测试选择性冻结 / Test selective freeze"""
        model = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2))
        # Freeze first layer only
        for param in model[0].parameters():
            param.stop_gradient = True
        # First layer frozen
        for param in model[0].parameters():
            self.assertTrue(param.stop_gradient)
        # Second layer not frozen
        for param in model[1].parameters():
            self.assertFalse(param.stop_gradient)

    def test_parameter_count_with_frozen(self):
        """测试带冻结参数的训练 / Test training with frozen parameters"""
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        # Freeze first linear
        for param in model[0].parameters():
            param.stop_gradient = True
        # Only second linear should have trainable params
        trainable = [p for p in model.parameters() if not p.stop_gradient]
        # Linear(8,2): 8*2+2=18 trainable params
        self.assertEqual(sum(p.numel() for p in trainable), 18)


class TestModelClone(unittest.TestCase):
    """测试模型克隆 / Test model cloning"""

    def test_model_copy(self):
        """测试模型复制 / Test model copy"""
        import copy

        model1 = nn.Linear(4, 2)
        model2 = copy.deepcopy(model1)

        # Verify weights are the same
        np.testing.assert_allclose(model1.weight.numpy(), model2.weight.numpy())

        # Modify model2 and verify model1 is unchanged
        with paddle.no_grad():
            model2.weight[:] = paddle.zeros_like(model2.weight)

        self.assertFalse(
            np.allclose(model1.weight.numpy(), model2.weight.numpy())
        )

    def test_sequential_access(self):
        """测试Sequential层访问 / Test Sequential layer access"""
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        # Access layers by index
        first_layer = model[0]
        self.assertIsInstance(first_layer, nn.Linear)
        last_layer = model[-1]
        self.assertIsInstance(last_layer, nn.Linear)


if __name__ == '__main__':
    unittest.main()
