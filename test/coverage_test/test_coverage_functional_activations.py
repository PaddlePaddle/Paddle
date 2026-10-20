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
激活函数单元测试 / Activation Function Unit Tests

测试目标 / Test Target:
  paddle.nn.functional 激活函数

覆盖的模块 / Covered Modules:
  - F.relu/relu6/leaky_relu/prelu
  - F.sigmoid/tanh/swish/silu
  - F.gelu/elu/selu/celu
  - F.softplus/softsign/mish
  - F.log_softmax/log_sigmoid

作用 / Purpose:
  补充函数式激活函数API的测试，提升覆盖率。
"""

import unittest

import numpy as np

import paddle
import paddle.nn.functional as F

paddle.disable_static()


class TestReluFamily(unittest.TestCase):
    """测试ReLU系列激活函数 / Test ReLU family activations"""

    def test_relu(self):
        """测试ReLU / Test ReLU"""
        x = paddle.to_tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = F.relu(x)
        np.testing.assert_allclose(result.numpy(), [0.0, 0.0, 0.0, 1.0, 2.0])

    def test_relu6(self):
        """测试ReLU6 / Test ReLU6"""
        x = paddle.to_tensor([-1.0, 0.0, 3.0, 6.0, 8.0])
        result = F.relu6(x)
        np.testing.assert_allclose(result.numpy(), [0.0, 0.0, 3.0, 6.0, 6.0])

    def test_leaky_relu(self):
        """测试LeakyReLU / Test LeakyReLU"""
        x = paddle.to_tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = F.leaky_relu(x, negative_slope=0.1)
        np.testing.assert_allclose(
            result.numpy(), [-0.2, -0.1, 0.0, 1.0, 2.0], rtol=1e-5
        )

    def test_elu(self):
        """测试ELU / Test ELU"""
        x = paddle.to_tensor([-1.0, 0.0, 1.0])
        result = F.elu(x, alpha=1.0)
        self.assertEqual(result.shape, [3])
        # For x < 0: alpha*(exp(x)-1), for x >= 0: x
        np.testing.assert_allclose(
            float(result[0].numpy()), 1.0 * (np.exp(-1.0) - 1), rtol=1e-5
        )

    def test_selu(self):
        """测试SELU / Test SELU"""
        x = paddle.to_tensor([-1.0, 0.0, 1.0])
        result = F.selu(x)
        self.assertEqual(result.shape, [3])

    def test_prelu(self):
        """测试PReLU / Test PReLU"""
        weight = paddle.to_tensor([0.25])
        x = paddle.to_tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = F.prelu(x, weight)
        np.testing.assert_allclose(
            result.numpy(), [-0.5, -0.25, 0.0, 1.0, 2.0], rtol=1e-5
        )


class TestSigmoidFamily(unittest.TestCase):
    """测试Sigmoid系列激活函数 / Test Sigmoid family activations"""

    def test_sigmoid(self):
        """测试Sigmoid / Test Sigmoid"""
        x = paddle.to_tensor([0.0])
        result = F.sigmoid(x)
        self.assertAlmostEqual(float(result.item()), 0.5, places=5)

    def test_tanh(self):
        """测试Tanh / Test Tanh"""
        x = paddle.to_tensor([0.0, 1.0, -1.0])
        result = F.tanh(x)
        np.testing.assert_allclose(
            result.numpy(), np.tanh([0.0, 1.0, -1.0]), rtol=1e-5
        )

    def test_hardtanh(self):
        """测试HardTanh / Test HardTanh"""
        x = paddle.to_tensor([-3.0, -0.5, 0.0, 0.5, 3.0])
        result = F.hardtanh(x, min=-1.0, max=1.0)
        np.testing.assert_allclose(result.numpy(), [-1.0, -0.5, 0.0, 0.5, 1.0])

    def test_hardsigmoid(self):
        """测试HardSigmoid / Test HardSigmoid"""
        x = paddle.to_tensor([-3.0, 0.0, 3.0])
        result = F.hardsigmoid(x)
        self.assertEqual(result.shape, [3])
        self.assertAlmostEqual(float(result[0].numpy()), 0.0, places=5)
        self.assertAlmostEqual(float(result[2].numpy()), 1.0, places=5)

    def test_log_sigmoid(self):
        """测试LogSigmoid / Test LogSigmoid"""
        x = paddle.to_tensor([0.0, 1.0, -1.0])
        result = F.log_sigmoid(x)
        expected = np.log(1 / (1 + np.exp(-np.array([0.0, 1.0, -1.0]))))
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)


class TestSwishAndMish(unittest.TestCase):
    """测试Swish/Mish激活函数 / Test Swish/Mish activations"""

    def test_swish(self):
        """测试Swish / Test Swish"""
        x = paddle.to_tensor([0.0, 1.0, -1.0])
        result = F.swish(x)
        self.assertEqual(result.shape, [3])
        # swish(x) = x * sigmoid(x)
        expected = np.array([0.0, 1.0, -1.0]) * (
            1 / (1 + np.exp(-np.array([0.0, 1.0, -1.0])))
        )
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

    def test_silu(self):
        """测试SiLU (Swish) / Test SiLU"""
        x = paddle.to_tensor([0.0, 1.0, -1.0])
        result = F.silu(x)
        self.assertEqual(result.shape, [3])

    def test_mish(self):
        """测试Mish / Test Mish"""
        x = paddle.to_tensor([0.0, 1.0, -1.0])
        result = F.mish(x)
        self.assertEqual(result.shape, [3])

    def test_gelu(self):
        """测试GELU / Test GELU"""
        x = paddle.to_tensor([0.0, 1.0, -1.0])
        result = F.gelu(x)
        self.assertEqual(result.shape, [3])

    def test_gelu_approximate(self):
        """测试近似GELU / Test approximate GELU"""
        x = paddle.to_tensor([0.0, 1.0, -1.0])
        result = F.gelu(x, approximate=True)
        self.assertEqual(result.shape, [3])


class TestSoftmaxFamily(unittest.TestCase):
    """测试Softmax系列 / Test Softmax family"""

    def test_softmax(self):
        """测试Softmax / Test Softmax"""
        x = paddle.to_tensor([[1.0, 2.0, 3.0]])
        result = F.softmax(x, axis=1)
        self.assertAlmostEqual(float(paddle.sum(result).numpy()), 1.0, places=5)

    def test_log_softmax(self):
        """测试LogSoftmax / Test LogSoftmax"""
        x = paddle.to_tensor([[1.0, 2.0, 3.0]])
        result = F.log_softmax(x, axis=1)
        # exp(log_softmax) should sum to 1
        np.testing.assert_allclose(
            float(paddle.exp(result).sum().numpy()), 1.0, rtol=1e-5
        )

    def test_softplus(self):
        """测试Softplus / Test Softplus"""
        x = paddle.to_tensor([-1.0, 0.0, 1.0])
        result = F.softplus(x)
        self.assertEqual(result.shape, [3])
        # All outputs should be positive
        self.assertTrue(bool((result > 0).all().numpy()))

    def test_softsign(self):
        """测试Softsign / Test Softsign"""
        x = paddle.to_tensor([-1.0, 0.0, 1.0])
        result = F.softsign(x)
        # softsign(x) = x / (1 + |x|)
        expected = np.array([-1.0, 0.0, 1.0]) / (1 + np.abs([-1.0, 0.0, 1.0]))
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
