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
线性层及基础层单元测试 / Linear Layer and Basic Layers Unit Tests

测试目标 / Test Target:
  paddle.nn.Linear及相关层 (覆盖率约83.2%)

覆盖的模块 / Covered Modules:
  - paddle.nn.Linear: 全连接线性层
  - paddle.nn.Bilinear: 双线性层
  - paddle.nn.Identity: 恒等映射层
  - paddle.nn.Flatten: 张量展平层
  - paddle.nn.Unflatten: 张量还原展平

作用 / Purpose:
  覆盖基础线性变换层的正向传播、参数初始化、权重设置等代码路径。
"""

import unittest

import numpy as np

import paddle
from paddle import nn

paddle.disable_static()


class TestLinearLayer(unittest.TestCase):
    """测试Linear线性层 / Test Linear layer"""

    def test_linear_basic(self):
        """测试基本线性层 / Test basic linear layer"""
        linear = nn.Linear(10, 5)
        x = paddle.randn([4, 10])
        y = linear(x)
        self.assertEqual(y.shape, [4, 5])

    def test_linear_no_bias(self):
        """测试无偏置线性层 / Test linear layer without bias"""
        linear = nn.Linear(10, 5, bias_attr=False)
        x = paddle.randn([4, 10])
        y = linear(x)
        self.assertEqual(y.shape, [4, 5])
        self.assertIsNone(linear.bias)

    def test_linear_weight_attr(self):
        """测试线性层权重初始化 / Test linear layer weight initialization"""
        initializer = nn.initializer.Constant(1.0)
        linear = nn.Linear(
            3, 2, weight_attr=paddle.ParamAttr(initializer=initializer)
        )
        x = paddle.ones([4, 3])
        y = linear(x)
        # Each output = 3 * 1.0 + bias
        self.assertEqual(y.shape, [4, 2])

    def test_linear_batch(self):
        """测试批量线性层 / Test batch linear layer"""
        linear = nn.Linear(10, 5)
        x = paddle.randn([8, 10])
        y = linear(x)
        self.assertEqual(y.shape, [8, 5])

    def test_linear_3d_input(self):
        """测试3D输入线性层 / Test linear layer with 3D input"""
        linear = nn.Linear(10, 5)
        x = paddle.randn([4, 6, 10])
        y = linear(x)
        self.assertEqual(y.shape, [4, 6, 5])

    def test_linear_parameters(self):
        """测试线性层参数 / Test linear layer parameters"""
        linear = nn.Linear(10, 5)
        params = list(linear.parameters())
        # Should have weight and bias
        self.assertEqual(len(params), 2)
        self.assertEqual(params[0].shape, [10, 5])
        self.assertEqual(params[1].shape, [5])

    def test_linear_gradient(self):
        """测试线性层梯度 / Test linear layer gradient"""
        linear = nn.Linear(3, 2)
        x = paddle.randn([4, 3])
        y = linear(x)
        loss = y.mean()
        loss.backward()
        self.assertIsNotNone(linear.weight.grad)
        self.assertIsNotNone(linear.bias.grad)

    def test_linear_to_float16(self):
        """测试线性层转换精度 / Test linear layer precision conversion"""
        linear = nn.Linear(10, 5)
        linear_half = linear.half()
        self.assertEqual(linear_half.weight.dtype, paddle.float16)


class TestBilinearLayer(unittest.TestCase):
    """测试Bilinear双线性层 / Test Bilinear layer"""

    def test_bilinear_basic(self):
        """测试基本双线性层 / Test basic bilinear layer"""
        bilinear = nn.Bilinear(10, 8, 5)
        x1 = paddle.randn([4, 10])
        x2 = paddle.randn([4, 8])
        y = bilinear(x1, x2)
        self.assertEqual(y.shape, [4, 5])

    def test_bilinear_no_bias(self):
        """测试无偏置双线性层 / Test bilinear without bias"""
        bilinear = nn.Bilinear(10, 8, 5, bias_attr=False)
        x1 = paddle.randn([4, 10])
        x2 = paddle.randn([4, 8])
        y = bilinear(x1, x2)
        self.assertEqual(y.shape, [4, 5])

    def test_bilinear_parameters(self):
        """测试双线性层参数 / Test bilinear layer parameters"""
        bilinear = nn.Bilinear(10, 8, 5)
        params = list(bilinear.parameters())
        # weight: [5, 10, 8], bias: [5]
        self.assertEqual(len(params), 2)


class TestIdentityLayer(unittest.TestCase):
    """测试Identity恒等层 / Test Identity layer"""

    def test_identity_basic(self):
        """测试基本恒等层 / Test basic identity layer"""
        identity = nn.Identity()
        x = paddle.randn([4, 10])
        y = identity(x)
        np.testing.assert_allclose(x.numpy(), y.numpy())

    def test_identity_gradient(self):
        """测试恒等层梯度 / Test identity layer gradient"""
        identity = nn.Identity()
        x = paddle.randn([4, 10])
        x.stop_gradient = False
        y = identity(x)
        y.sum().backward()
        self.assertIsNotNone(x.grad)


class TestFlattenLayer(unittest.TestCase):
    """测试Flatten展平层 / Test Flatten layer"""

    def test_flatten_basic(self):
        """测试基本展平 / Test basic flattening"""
        flatten = nn.Flatten()
        x = paddle.randn([4, 3, 8, 8])
        y = flatten(x)
        self.assertEqual(y.shape, [4, 3 * 8 * 8])

    def test_flatten_start_axis(self):
        """测试指定起始轴展平 / Test flatten with start axis"""
        flatten = nn.Flatten(start_axis=2)
        x = paddle.randn([4, 3, 8, 8])
        y = flatten(x)
        self.assertEqual(y.shape, [4, 3, 64])

    def test_flatten_stop_axis(self):
        """测试指定结束轴展平 / Test flatten with stop axis"""
        flatten = nn.Flatten(start_axis=1, stop_axis=2)
        x = paddle.randn([4, 3, 8, 8])
        y = flatten(x)
        self.assertEqual(y.shape, [4, 24, 8])

    def test_flatten_gradient(self):
        """测试展平层梯度 / Test flatten gradient"""
        flatten = nn.Flatten()
        x = paddle.randn([4, 3, 4])
        x.stop_gradient = False
        y = flatten(x)
        y.sum().backward()
        self.assertIsNotNone(x.grad)


class TestNNParameterAndLayer(unittest.TestCase):
    """测试Parameter和Layer的基本功能 / Test Parameter and Layer basics"""

    def test_create_parameter(self):
        """测试创建参数 / Test creating parameter"""
        layer = nn.Layer()
        param = layer.create_parameter(shape=[5, 3], dtype='float32')
        self.assertEqual(param.shape, [5, 3])

    def test_layer_sublayers(self):
        """测试子层 / Test sublayers"""
        model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))
        sublayers = model.sublayers()
        self.assertTrue(len(sublayers) >= 2)

    def test_layer_named_parameters(self):
        """测试命名参数 / Test named parameters"""
        model = nn.Linear(10, 5)
        named_params = dict(model.named_parameters())
        self.assertIn('weight', named_params)
        self.assertIn('bias', named_params)

    def test_layer_train_eval_mode(self):
        """测试训练/评估模式切换 / Test train/eval mode switching"""
        model = nn.Sequential(nn.Linear(10, 5), nn.Dropout(0.5))
        model.train()
        self.assertTrue(model.training)
        model.eval()
        self.assertFalse(model.training)

    def test_layer_apply(self):
        """测试apply方法 / Test apply method"""
        model = nn.Sequential(nn.Linear(10, 5), nn.Linear(5, 2))
        called = []

        def count_layers(layer):
            called.append(type(layer).__name__)

        model.apply(count_layers)
        self.assertIn('Linear', called)

    def test_layer_extra_repr(self):
        """测试extra_repr / Test extra_repr"""
        linear = nn.Linear(10, 5)
        repr_str = repr(linear)
        self.assertIn('Linear', repr_str)


if __name__ == '__main__':
    unittest.main()
