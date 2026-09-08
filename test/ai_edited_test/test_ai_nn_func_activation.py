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

# [AUTO-GENERATED]
# Target: paddle/nn/functional/activation.py
# Coverage target: improve coverage for activation functions (celu, elu, hardshrink, hardtanh,
#   hardsigmoid, hardswish, leaky_relu, prelu, rrelu, relu, log_sigmoid, maxout, relu6,
#   selu, silu, softmax, softshrink, softsign, swish, mish, tanhshrink, thresholded_relu,
#   log_softmax, glu, gumbel_softmax, swiglu)
"""
Tests for paddle.nn.functional.activation module.
测试 paddle.nn.functional.activation 模块的单元测试。
"""

import unittest

import numpy as np

import paddle
from paddle.nn import functional as F


class TestCelu(unittest.TestCase):
    """Tests for celu activation function. / celu 激活函数的测试。"""

    def setUp(self):
        self.x = paddle.to_tensor([-1.0, 0.0, 1.0, 2.0], dtype='float32')

    def test_celu_default(self):
        """Test celu with default alpha. / 测试默认 alpha 参数的 celu。"""
        out = F.celu(self.x)
        self.assertEqual(out.shape, self.x.shape)

    def test_celu_alpha(self):
        """Test celu with custom alpha. / 测试自定义 alpha 参数的 celu。"""
        out = F.celu(self.x, alpha=0.5)
        self.assertEqual(out.shape, self.x.shape)

    def test_celu_zero_alpha_error(self):
        """Test celu raises ZeroDivisionError for alpha=0. / 测试 alpha=0 时 celu 抛出 ZeroDivisionError。"""
        with self.assertRaises(ZeroDivisionError):
            F.celu(self.x, alpha=0)


class TestElu(unittest.TestCase):
    """Tests for elu activation function. / elu 激活函数的测试。"""

    def setUp(self):
        self.x = paddle.to_tensor([-1.0, 0.0, 1.0, 2.0], dtype='float32')

    def test_elu_default(self):
        """Test elu with default alpha. / 测试默认 alpha 参数的 elu。"""
        out = F.elu(self.x)
        self.assertEqual(out.shape, self.x.shape)

    def test_elu_custom_alpha(self):
        """Test elu with custom alpha. / 测试自定义 alpha 参数的 elu。"""
        out = F.elu(self.x, alpha=0.5)
        self.assertEqual(out.shape, self.x.shape)


class TestHardshrink(unittest.TestCase):
    """Tests for hardshrink activation function. / hardshrink 激活函数的测试。"""

    def setUp(self):
        self.x = paddle.to_tensor([-1.0, -0.3, 0.3, 1.0], dtype='float32')

    def test_hardshrink_default(self):
        """Test hardshrink with default threshold. / 测试默认阈值的 hardshrink。"""
        out = F.hardshrink(self.x)
        self.assertEqual(out.shape, self.x.shape)

    def test_hardshrink_custom_threshold(self):
        """Test hardshrink with custom threshold. / 测试自定义阈值的 hardshrink。"""
        out = F.hardshrink(self.x, threshold=0.2)
        self.assertEqual(out.shape, self.x.shape)

    def test_hardshrink_alias_lambd(self):
        """Test hardshrink with lambd alias. / 测试 lambd 别名的 hardshrink。"""
        out = F.hardshrink(self.x, lambd=0.2)
        self.assertEqual(out.shape, self.x.shape)


class TestHardtanh(unittest.TestCase):
    """Tests for hardtanh activation function. / hardtanh 激活函数的测试。"""

    def setUp(self):
        self.x = paddle.to_tensor([-2.0, -0.5, 0.5, 2.0], dtype='float32')

    def test_hardtanh_default(self):
        """Test hardtanh with default range. / 测试默认范围的 hardtanh。"""
        out = F.hardtanh(self.x)
        self.assertEqual(out.shape, self.x.shape)

    def test_hardtanh_custom_range(self):
        """Test hardtanh with custom min/max. / 测试自定义 min/max 的 hardtanh。"""
        out = F.hardtanh(self.x, min=-0.5, max=0.5)
        result = out.numpy()
        np.testing.assert_allclose(result, [-0.5, -0.5, 0.5, 0.5], rtol=1e-5)


class TestHardsigmoid(unittest.TestCase):
    """Tests for hardsigmoid activation function. / hardsigmoid 激活函数的测试。"""

    def setUp(self):
        self.x = paddle.to_tensor([-4.0, 0.0, 3.0, 5.0], dtype='float32')

    def test_hardsigmoid_default(self):
        """Test hardsigmoid with default params. / 测试默认参数的 hardsigmoid。"""
        out = F.hardsigmoid(self.x)
        result = out.numpy()
        np.testing.assert_allclose(result[0], 0.0, rtol=1e-5)
        np.testing.assert_allclose(result[2], 1.0, rtol=1e-5)

    def test_hardsigmoid_custom_params(self):
        """Test hardsigmoid with custom slope/offset. / 测试自定义 slope/offset 的 hardsigmoid。"""
        out = F.hardsigmoid(self.x, slope=0.2, offset=0.4)
        self.assertEqual(out.shape, self.x.shape)


class TestHardswish(unittest.TestCase):
    """Tests for hardswish activation function. / hardswish 激活函数的测试。"""

    def setUp(self):
        self.x = paddle.to_tensor([-4.0, 0.0, 3.0, 5.0], dtype='float32')

    def test_hardswish(self):
        """Test hardswish activation. / 测试 hardswish 激活。"""
        out = F.hardswish(self.x)
        self.assertEqual(out.shape, self.x.shape)
        # x <= -3 => 0
        np.testing.assert_allclose(out.numpy()[0], 0.0, atol=1e-6)
        # x >= 3 => x
        np.testing.assert_allclose(out.numpy()[3], 5.0, atol=1e-6)


class TestLeakyRelu(unittest.TestCase):
    """Tests for leaky_relu activation function. / leaky_relu 激活函数的测试。"""

    def setUp(self):
        self.x = paddle.to_tensor([-2.0, 0.0, 1.0], dtype='float32')

    def test_leaky_relu_default(self):
        """Test leaky_relu with default negative_slope. / 测试默认 negative_slope 的 leaky_relu。"""
        out = F.leaky_relu(self.x)
        self.assertEqual(out.shape, self.x.shape)

    def test_leaky_relu_custom_slope(self):
        """Test leaky_relu with custom slope. / 测试自定义斜率的 leaky_relu。"""
        out = F.leaky_relu(self.x, negative_slope=0.1)
        result = out.numpy()
        self.assertAlmostEqual(result[0], -0.2, places=5)
        self.assertAlmostEqual(result[1], 0.0, places=5)
        self.assertAlmostEqual(result[2], 1.0, places=5)

    def test_leaky_relu_alias_input(self):
        """Test leaky_relu with input alias. / 测试 input 别名的 leaky_relu。"""
        out = F.leaky_relu(input=self.x)
        self.assertEqual(out.shape, self.x.shape)


class TestPrelu(unittest.TestCase):
    """Tests for prelu activation function. / prelu 激活函数的测试。"""

    def test_prelu_scalar_weight(self):
        """Test prelu with scalar weight. / 测试标量权重的 prelu。"""
        x = paddle.to_tensor([-1.0, 0.0, 1.0], dtype='float32')
        w = paddle.to_tensor([0.25], dtype='float32')
        out = F.prelu(x, w)
        self.assertEqual(out.shape, x.shape)

    def test_prelu_channel_weight(self):
        """Test prelu with per-channel weight. / 测试通道权重的 prelu。"""
        x = paddle.randn([2, 3, 4, 4], dtype='float32')
        w = paddle.to_tensor([0.1, 0.2, 0.3], dtype='float32')
        out = F.prelu(x, w)
        self.assertEqual(out.shape, x.shape)

    def test_prelu_nhwc_weight(self):
        """Test prelu with NHWC data format. / 测试 NHWC 格式的 prelu。"""
        x = paddle.randn([2, 4, 4, 3], dtype='float32')
        w = paddle.to_tensor([0.1, 0.2, 0.3], dtype='float32')
        out = F.prelu(x, w, data_format='NHWC')
        self.assertEqual(out.shape, x.shape)

    def test_prelu_invalid_data_format(self):
        """Test prelu raises ValueError for invalid data_format. / 测试无效 data_format 时 prelu 抛出 ValueError。"""
        x = paddle.randn([2, 3, 4, 4], dtype='float32')
        w = paddle.to_tensor([0.1, 0.2, 0.3], dtype='float32')
        with self.assertRaises(ValueError):
            F.prelu(x, w, data_format='invalid')


class TestRrelu(unittest.TestCase):
    """Tests for rrelu activation function. / rrelu 激活函数的测试。"""

    def setUp(self):
        self.x = paddle.to_tensor([-2.0, 0.0, 1.0], dtype='float32')

    def test_rrelu_training(self):
        """Test rrelu in training mode. / 测试训练模式下的 rrelu。"""
        out = F.rrelu(self.x, lower=0.1, upper=0.3, training=True)
        self.assertEqual(out.shape, self.x.shape)

    def test_rrelu_eval(self):
        """Test rrelu in eval mode. / 测试评估模式下的 rrelu。"""
        out = F.rrelu(self.x, lower=0.1, upper=0.3, training=False)
        self.assertEqual(out.shape, self.x.shape)

    def test_rrelu_invalid_lower(self):
        """Test rrelu raises ValueError for invalid lower. / 测试无效 lower 时 rrelu 抛出 ValueError。"""
        with self.assertRaises(ValueError):
            F.rrelu(self.x, lower=1.5, upper=0.3)

    def test_rrelu_upper_less_than_lower(self):
        """Test rrelu raises ValueError when upper < lower. / 测试 upper < lower 时 rrelu 抛出 ValueError。"""
        with self.assertRaises(ValueError):
            F.rrelu(self.x, lower=0.3, upper=0.1)

    def test_rrelu_upper_greater_than_one(self):
        """Test rrelu raises ValueError when upper > 1. / 测试 upper > 1 时 rrelu 抛出 ValueError。"""
        with self.assertRaises(ValueError):
            F.rrelu(self.x, lower=0.1, upper=1.5)

    def test_rrelu_invalid_type(self):
        """Test rrelu raises TypeError for non-float bounds. / 测试非浮点数边界时 rrelu 抛出 TypeError。"""
        with self.assertRaises(TypeError):
            F.rrelu(self.x, lower="0.1", upper=0.3)


class TestRelu(unittest.TestCase):
    """Tests for relu activation function. / relu 激活函数的测试。"""

    def setUp(self):
        self.x = paddle.to_tensor([-2.0, 0.0, 1.0], dtype='float32')

    def test_relu(self):
        """Test relu activation. / 测试 relu 激活。"""
        out = F.relu(self.x)
        result = out.numpy()
        np.testing.assert_allclose(result, [0.0, 0.0, 1.0], rtol=1e-5)

    def test_relu_alias_input(self):
        """Test relu with input alias. / 测试 input 别名的 relu。"""
        out = F.relu(input=self.x)
        self.assertEqual(out.shape, self.x.shape)


class TestLogSigmoid(unittest.TestCase):
    """Tests for log_sigmoid activation function. / log_sigmoid 激活函数的测试。"""

    def test_log_sigmoid(self):
        """Test log_sigmoid activation. / 测试 log_sigmoid 激活。"""
        x = paddle.to_tensor([0.0, 1.0, 2.0], dtype='float32')
        out = F.log_sigmoid(x)
        self.assertEqual(out.shape, x.shape)

    def test_log_sigmoid_alias_input(self):
        """Test log_sigmoid with input alias. / 测试 input 别名的 log_sigmoid。"""
        x = paddle.to_tensor([0.0, 1.0], dtype='float32')
        out = F.log_sigmoid(input=x)
        self.assertEqual(out.shape, x.shape)


class TestMaxout(unittest.TestCase):
    """Tests for maxout activation function. / maxout 激活函数的测试。"""

    def test_maxout_axis1(self):
        """Test maxout with axis=1. / 测试 axis=1 的 maxout。"""
        x = paddle.randn([1, 4, 2, 2], dtype='float32')
        out = F.maxout(x, groups=2, axis=1)
        self.assertEqual(out.shape, [1, 2, 2, 2])

    def test_maxout_axis3(self):
        """Test maxout with axis=3. / 测试 axis=3 的 maxout。"""
        x = paddle.randn([1, 2, 2, 4], dtype='float32')
        out = F.maxout(x, groups=2, axis=3)
        self.assertEqual(out.shape, [1, 2, 2, 2])

    def test_maxout_axis_negative1(self):
        """Test maxout with axis=-1. / 测试 axis=-1 的 maxout。"""
        x = paddle.randn([1, 2, 2, 4], dtype='float32')
        out = F.maxout(x, groups=2, axis=-1)
        self.assertEqual(out.shape, [1, 2, 2, 2])


class TestRelu6(unittest.TestCase):
    """Tests for relu6 activation function. / relu6 激活函数的测试。"""

    def test_relu6(self):
        """Test relu6 activation. / 测试 relu6 激活。"""
        x = paddle.to_tensor([-1.0, 0.0, 3.0, 7.0], dtype='float32')
        out = F.relu6(x)
        result = out.numpy()
        np.testing.assert_allclose(result, [0.0, 0.0, 3.0, 6.0], rtol=1e-5)


class TestSelu(unittest.TestCase):
    """Tests for selu activation function. / selu 激活函数的测试。"""

    def test_selu_default(self):
        """Test selu with default params. / 测试默认参数的 selu。"""
        x = paddle.to_tensor([0.0, 1.0, -1.0], dtype='float32')
        out = F.selu(x)
        self.assertEqual(out.shape, x.shape)

    def test_selu_invalid_scale(self):
        """Test selu raises ValueError for invalid scale. / 测试无效 scale 时 selu 抛出 ValueError。"""
        x = paddle.to_tensor([0.0], dtype='float32')
        with self.assertRaises(ValueError):
            F.selu(x, scale=0.5)

    def test_selu_invalid_alpha(self):
        """Test selu raises ValueError for invalid alpha. / 测试无效 alpha 时 selu 抛出 ValueError。"""
        x = paddle.to_tensor([0.0], dtype='float32')
        with self.assertRaises(ValueError):
            F.selu(x, alpha=-1.0)


class TestSilu(unittest.TestCase):
    """Tests for silu activation function. / silu 激活函数的测试。"""

    def test_silu(self):
        """Test silu activation. / 测试 silu 激活。"""
        x = paddle.to_tensor([0.0, 1.0, 2.0], dtype='float32')
        out = F.silu(x)
        self.assertEqual(out.shape, x.shape)

    def test_silu_alias_input(self):
        """Test silu with input alias. / 测试 input 别名的 silu。"""
        x = paddle.to_tensor([0.0, 1.0], dtype='float32')
        out = F.silu(input=x)
        self.assertEqual(out.shape, x.shape)


class TestSoftmax(unittest.TestCase):
    """Tests for softmax activation function. / softmax 激活函数的测试。"""

    def test_softmax_default(self):
        """Test softmax with default axis. / 测试默认 axis 的 softmax。"""
        x = paddle.randn([2, 3, 4], dtype='float32')
        out = F.softmax(x)
        self.assertEqual(out.shape, x.shape)
        # Sum along last axis should be 1
        s = out.sum(axis=-1).numpy()
        np.testing.assert_allclose(s, 1.0, rtol=1e-5)

    def test_softmax_axis0(self):
        """Test softmax with axis=0. / 测试 axis=0 的 softmax。"""
        x = paddle.randn([2, 3, 4], dtype='float32')
        out = F.softmax(x, axis=0)
        self.assertEqual(out.shape, x.shape)

    def test_softmax_with_dtype(self):
        """Test softmax with dtype conversion. / 测试带 dtype 转换的 softmax。"""
        x = paddle.randn([2, 3], dtype='float32')
        out = F.softmax(x, dtype='float64')
        self.assertEqual(out.dtype, paddle.float64)

    def test_softmax_alias_dim(self):
        """Test softmax with dim alias. / 测试 dim 别名的 softmax。"""
        x = paddle.randn([2, 3], dtype='float32')
        out = F.softmax(x, dim=0)
        self.assertEqual(out.shape, x.shape)


class TestSoftshrink(unittest.TestCase):
    """Tests for softshrink activation function. / softshrink 激活函数的测试。"""

    def test_softshrink_default(self):
        """Test softshrink with default threshold. / 测试默认阈值的 softshrink。"""
        x = paddle.to_tensor([-1.0, -0.3, 0.3, 1.0], dtype='float32')
        out = F.softshrink(x)
        self.assertEqual(out.shape, x.shape)

    def test_softshrink_negative_threshold(self):
        """Test softshrink raises ValueError for negative threshold. / 测试负阈值时 softshrink 抛出 ValueError。"""
        x = paddle.to_tensor([0.0], dtype='float32')
        with self.assertRaises(ValueError):
            F.softshrink(x, threshold=-0.1)

    def test_softshrink_alias_lambd(self):
        """Test softshrink with lambd alias. / 测试 lambd 别名的 softshrink。"""
        x = paddle.to_tensor([-1.0, 1.0], dtype='float32')
        out = F.softshrink(x, lambd=0.5)
        self.assertEqual(out.shape, x.shape)


class TestSoftsign(unittest.TestCase):
    """Tests for softsign activation function. / softsign 激活函数的测试。"""

    def test_softsign(self):
        """Test softsign activation. / 测试 softsign 激活。"""
        x = paddle.to_tensor([-1.0, 0.0, 1.0], dtype='float32')
        out = F.softsign(x)
        self.assertEqual(out.shape, x.shape)
        np.testing.assert_allclose(out.numpy()[1], 0.0, rtol=1e-5)


class TestSwish(unittest.TestCase):
    """Tests for swish activation function. / swish 激活函数的测试。"""

    def test_swish(self):
        """Test swish activation. / 测试 swish 激活。"""
        x = paddle.to_tensor([0.0, 1.0, -1.0], dtype='float32')
        out = F.swish(x)
        self.assertEqual(out.shape, x.shape)


class TestMish(unittest.TestCase):
    """Tests for mish activation function. / mish 激活函数的测试。"""

    def test_mish(self):
        """Test mish activation. / 测试 mish 激活。"""
        x = paddle.to_tensor([-1.0, 0.0, 1.0], dtype='float32')
        out = F.mish(x)
        self.assertEqual(out.shape, x.shape)

    def test_mish_alias_input(self):
        """Test mish with input alias. / 测试 input 别名的 mish。"""
        x = paddle.to_tensor([0.0, 1.0], dtype='float32')
        out = F.mish(input=x)
        self.assertEqual(out.shape, x.shape)


class TestTanhshrink(unittest.TestCase):
    """Tests for tanhshrink activation function. / tanhshrink 激活函数的测试。"""

    def test_tanhshrink(self):
        """Test tanhshrink activation. / 测试 tanhshrink 激活。"""
        x = paddle.to_tensor([-1.0, 0.0, 1.0], dtype='float32')
        out = F.tanhshrink(x)
        self.assertEqual(out.shape, x.shape)


class TestThresholdedRelu(unittest.TestCase):
    """Tests for thresholded_relu activation function. / thresholded_relu 激活函数的测试。"""

    def test_thresholded_relu_default(self):
        """Test thresholded_relu with default params. / 测试默认参数的 thresholded_relu。"""
        x = paddle.to_tensor([0.5, 1.5, 2.5], dtype='float32')
        out = F.thresholded_relu(x)
        result = out.numpy()
        np.testing.assert_allclose(result, [0.0, 1.5, 2.5], rtol=1e-5)

    def test_thresholded_relu_custom(self):
        """Test thresholded_relu with custom threshold and value. / 测试自定义参数的 thresholded_relu。"""
        x = paddle.to_tensor([0.5, 1.5, 2.5], dtype='float32')
        out = F.thresholded_relu(x, threshold=2.0, value=-1.0)
        result = out.numpy()
        np.testing.assert_allclose(result, [-1.0, -1.0, 2.5], rtol=1e-5)


class TestLogSoftmax(unittest.TestCase):
    """Tests for log_softmax activation function. / log_softmax 激活函数的测试。"""

    def test_log_softmax_default(self):
        """Test log_softmax with default axis. / 测试默认 axis 的 log_softmax。"""
        x = paddle.randn([2, 3, 4], dtype='float32')
        out = F.log_softmax(x)
        self.assertEqual(out.shape, x.shape)

    def test_log_softmax_with_dtype(self):
        """Test log_softmax with dtype. / 测试带 dtype 的 log_softmax。"""
        x = paddle.randn([2, 3], dtype='float32')
        out = F.log_softmax(x, dtype='float64')
        self.assertEqual(out.dtype, paddle.float64)

    def test_log_softmax_alias_dim(self):
        """Test log_softmax with dim alias. / 测试 dim 别名的 log_softmax。"""
        x = paddle.randn([2, 3], dtype='float32')
        out = F.log_softmax(x, dim=0)
        self.assertEqual(out.shape, x.shape)


class TestGlu(unittest.TestCase):
    """Tests for glu activation function. / glu 激活函数的测试。"""

    def test_glu_default(self):
        """Test glu with default axis. / 测试默认 axis 的 glu。"""
        x = paddle.randn([2, 4], dtype='float32')
        out = F.glu(x)
        self.assertEqual(out.shape, [2, 2])

    def test_glu_axis0(self):
        """Test glu with axis=0. / 测试 axis=0 的 glu。"""
        x = paddle.randn([4, 2], dtype='float32')
        out = F.glu(x, axis=0)
        self.assertEqual(out.shape, [2, 2])

    def test_glu_invalid_axis(self):
        """Test glu raises ValueError for invalid axis. / 测试无效 axis 时 glu 抛出 ValueError。"""
        x = paddle.randn([2, 4], dtype='float32')
        with self.assertRaises(ValueError):
            F.glu(x, axis=10)

    def test_glu_alias_dim(self):
        """Test glu with dim alias. / 测试 dim 别名的 glu。"""
        x = paddle.randn([2, 4], dtype='float32')
        out = F.glu(x, dim=-1)
        self.assertEqual(out.shape, [2, 2])


class TestGumbelSoftmax(unittest.TestCase):
    """Tests for gumbel_softmax activation function. / gumbel_softmax 激活函数的测试。"""

    def test_gumbel_softmax_default(self):
        """Test gumbel_softmax with default params. / 测试默认参数的 gumbel_softmax。"""
        x = paddle.randn([2, 4], dtype='float32')
        out = F.gumbel_softmax(x)
        self.assertEqual(out.shape, x.shape)

    def test_gumbel_softmax_hard(self):
        """Test gumbel_softmax with hard=True. / 测试 hard=True 的 gumbel_softmax。"""
        paddle.seed(42)
        x = paddle.randn([2, 4], dtype='float32')
        out = F.gumbel_softmax(x, hard=True, temperature=0.1)
        self.assertEqual(out.shape, x.shape)

    def test_gumbel_softmax_temperature(self):
        """Test gumbel_softmax with custom temperature. / 测试自定义温度的 gumbel_softmax。"""
        x = paddle.randn([2, 4], dtype='float32')
        out = F.gumbel_softmax(x, temperature=0.5)
        self.assertEqual(out.shape, x.shape)


class TestSwiglu(unittest.TestCase):
    """Tests for swiglu activation function. / swiglu 激活函数的测试。"""

    def test_swiglu_single_input(self):
        """Test swiglu with single input (auto-chunk). / 测试单输入的 swiglu（自动分块）。"""
        x = paddle.to_tensor([1.0, 2.0, 3.0, 4.0], dtype='float32')
        out = F.swiglu(x)
        self.assertEqual(out.shape, [2])

    def test_swiglu_two_inputs(self):
        """Test swiglu with two inputs. / 测试双输入的 swiglu。"""
        x = paddle.to_tensor([1.0, 2.0], dtype='float32')
        y = paddle.to_tensor([3.0, 4.0], dtype='float32')
        out = F.swiglu(x, y)
        self.assertEqual(out.shape, [2])


if __name__ == '__main__':
    unittest.main()
