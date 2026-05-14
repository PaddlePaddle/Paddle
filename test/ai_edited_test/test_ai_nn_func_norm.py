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
# Target: paddle/nn/functional/norm.py
# Coverage target: improve coverage for norm functions (normalize, batch_norm, layer_norm,
#   instance_norm, local_response_norm, group_norm, rms_norm)
"""
Tests for paddle.nn.functional.norm module.
测试 paddle.nn.functional.norm 模块的单元测试。
"""

import unittest

import paddle
from paddle.nn import functional as F


class TestNormalize(unittest.TestCase):
    """Tests for normalize function. / normalize 函数的测试。"""

    def test_normalize_l2(self):
        """Test normalize with p=2. / 测试 p=2 的 normalize。"""
        x = paddle.randn([2, 4], dtype='float32')
        out = F.normalize(x, p=2.0, axis=1)
        self.assertEqual(out.shape, [2, 4])

    def test_normalize_l1(self):
        """Test normalize with p=1. / 测试 p=1 的 normalize。"""
        x = paddle.randn([2, 4], dtype='float32')
        out = F.normalize(x, p=1.0, axis=1)
        self.assertEqual(out.shape, [2, 4])

    def test_normalize_inf(self):
        """Test normalize with p=float('inf'). / 测试 p=inf 的 normalize。"""
        x = paddle.randn([2, 4], dtype='float32')
        out = F.normalize(x, p=float('inf'), axis=1)
        self.assertEqual(out.shape, [2, 4])

    def test_normalize_negative_axis(self):
        """Test normalize with negative axis. / 测试负 axis 的 normalize。"""
        x = paddle.randn([2, 4], dtype='float32')
        out = F.normalize(x, p=2.0, axis=-1)
        self.assertEqual(out.shape, [2, 4])

    def test_normalize_invalid_p(self):
        """Test normalize raises ValueError for invalid p. / 测试无效 p 时 normalize 抛出 ValueError。"""
        x = paddle.randn([2, 4], dtype='float32')
        with self.assertRaises(ValueError):
            F.normalize(x, p='invalid')


class TestBatchNorm(unittest.TestCase):
    """Tests for batch_norm function. / batch_norm 函数的测试。"""

    def setUp(self):
        self.x = paddle.randn([2, 3, 4, 4], dtype='float32')
        self.weight = paddle.randn([3], dtype='float32')
        self.bias = paddle.randn([3], dtype='float32')
        self.running_mean = paddle.zeros([3], dtype='float32')
        self.running_var = paddle.ones([3], dtype='float32')

    def test_batch_norm_train(self):
        """Test batch_norm in training mode. / 测试训练模式的 batch_norm。"""
        out = F.batch_norm(
            self.x,
            self.running_mean,
            self.running_var,
            weight=self.weight,
            bias=self.bias,
            training=True,
        )
        self.assertEqual(out.shape, [2, 3, 4, 4])

    def test_batch_norm_eval(self):
        """Test batch_norm in eval mode. / 测试评估模式的 batch_norm。"""
        out = F.batch_norm(
            self.x,
            self.running_mean,
            self.running_var,
            weight=self.weight,
            bias=self.bias,
            training=False,
        )
        self.assertEqual(out.shape, [2, 3, 4, 4])

    def test_batch_norm_no_weight_bias(self):
        """Test batch_norm without weight and bias. / 测试无权重偏置的 batch_norm。"""
        out = F.batch_norm(
            self.x, self.running_mean, self.running_var, training=True
        )
        self.assertEqual(out.shape, [2, 3, 4, 4])

    def test_batch_norm_1d(self):
        """Test batch_norm with 1D input. / 测试一维输入的 batch_norm。"""
        x1d = paddle.randn([4, 8], dtype='float32')
        rm = paddle.zeros([8], dtype='float32')
        rv = paddle.ones([8], dtype='float32')
        w = paddle.ones([8], dtype='float32')
        b = paddle.zeros([8], dtype='float32')
        out = F.batch_norm(x1d, rm, rv, weight=w, bias=b, training=True)
        self.assertEqual(out.shape, [4, 8])

    def test_batch_norm_3d(self):
        """Test batch_norm with 3D input. / 测试三维输入的 batch_norm。"""
        x3d = paddle.randn([2, 3, 8], dtype='float32')
        w3 = paddle.ones([3], dtype='float32')
        b3 = paddle.zeros([3], dtype='float32')
        out = F.batch_norm(
            x3d,
            self.running_mean[:3],
            self.running_var[:3],
            weight=w3,
            bias=b3,
            training=True,
        )
        self.assertEqual(out.shape, [2, 3, 8])

    def test_batch_norm_momentum(self):
        """Test batch_norm with custom momentum. / 测试自定义动量的 batch_norm。"""
        out = F.batch_norm(
            self.x,
            self.running_mean,
            self.running_var,
            weight=self.weight,
            bias=self.bias,
            training=True,
            momentum=0.1,
        )
        self.assertEqual(out.shape, [2, 3, 4, 4])

    def test_batch_norm_epsilon(self):
        """Test batch_norm with custom epsilon. / 测试自定义 epsilon 的 batch_norm。"""
        out = F.batch_norm(
            self.x,
            self.running_mean,
            self.running_var,
            weight=self.weight,
            bias=self.bias,
            training=True,
            epsilon=1e-3,
        )
        self.assertEqual(out.shape, [2, 3, 4, 4])

    def test_batch_norm_nhwc(self):
        """Test batch_norm with NHWC data format. / 测试 NHWC 格式的 batch_norm。"""
        x_nhwc = self.x.transpose([0, 2, 3, 1])
        out = F.batch_norm(
            x_nhwc,
            self.running_mean,
            self.running_var,
            weight=self.weight,
            bias=self.bias,
            training=True,
            data_format='NHWC',
        )
        self.assertIsNotNone(out)


class TestLayerNorm(unittest.TestCase):
    """Tests for layer_norm function. / layer_norm 函数的测试。"""

    def test_layer_norm_basic(self):
        """Test layer_norm with basic params. / 测试基本参数的 layer_norm。"""
        x = paddle.randn([2, 4, 8], dtype='float32')
        w = paddle.ones([8], dtype='float32')
        b = paddle.zeros([8], dtype='float32')
        out = F.layer_norm(x, w.shape, weight=w, bias=b)
        self.assertEqual(out.shape, [2, 4, 8])

    def test_layer_norm_no_weight_bias(self):
        """Test layer_norm without weight and bias. / 测试无权重偏置的 layer_norm。"""
        x = paddle.randn([2, 4, 8], dtype='float32')
        out = F.layer_norm(x, [8])
        self.assertEqual(out.shape, [2, 4, 8])

    def test_layer_norm_epsilon(self):
        """Test layer_norm with custom epsilon. / 测试自定义 epsilon 的 layer_norm。"""
        x = paddle.randn([2, 4, 8], dtype='float32')
        w = paddle.ones([8], dtype='float32')
        b = paddle.zeros([8], dtype='float32')
        out = F.layer_norm(x, [8], weight=w, bias=b, epsilon=1e-3)
        self.assertEqual(out.shape, [2, 4, 8])

    def test_layer_norm_2d(self):
        """Test layer_norm with 2D input. / 测试二维输入的 layer_norm。"""
        x = paddle.randn([4, 8], dtype='float32')
        w = paddle.ones([8], dtype='float32')
        b = paddle.zeros([8], dtype='float32')
        out = F.layer_norm(x, [8], weight=w, bias=b)
        self.assertEqual(out.shape, [4, 8])


class TestRmsNorm(unittest.TestCase):
    """Tests for rms_norm function. / rms_norm 函数的测试。"""

    def test_rms_norm_basic(self):
        """Test rms_norm with basic params. / 测试基本参数的 rms_norm。"""
        x = paddle.randn([2, 4, 8], dtype='float32')
        w = paddle.ones([8], dtype='float32')
        out = F.rms_norm(x, w.shape, weight=w)
        self.assertEqual(out.shape, [2, 4, 8])

    def test_rms_norm_epsilon(self):
        """Test rms_norm with custom epsilon. / 测试自定义 epsilon 的 rms_norm。"""
        x = paddle.randn([2, 4, 8], dtype='float32')
        w = paddle.ones([8], dtype='float32')
        out = F.rms_norm(x, w.shape, weight=w, eps=1e-4)
        self.assertEqual(out.shape, [2, 4, 8])


class TestInstanceNorm(unittest.TestCase):
    """Tests for instance_norm function. / instance_norm 函数的测试。"""

    def setUp(self):
        self.x = paddle.randn([2, 3, 4, 4], dtype='float32')
        self.weight = paddle.ones([3], dtype='float32')
        self.bias = paddle.zeros([3], dtype='float32')

    def test_instance_norm_basic(self):
        """Test instance_norm with basic params. / 测试基本参数的 instance_norm。"""
        out = F.instance_norm(self.x, weight=self.weight, bias=self.bias)
        self.assertEqual(out.shape, [2, 3, 4, 4])

    def test_instance_norm_no_weight_bias(self):
        """Test instance_norm without weight and bias. / 测试无权重偏置的 instance_norm。"""
        out = F.instance_norm(self.x)
        self.assertEqual(out.shape, [2, 3, 4, 4])

    def test_instance_norm_with_running_stats(self):
        """Test instance_norm with running stats. / 测试带运行统计的 instance_norm。"""
        running_mean = paddle.zeros([3], dtype='float32')
        running_var = paddle.ones([3], dtype='float32')
        out = F.instance_norm(
            self.x,
            running_mean,
            running_var,
            weight=self.weight,
            bias=self.bias,
            use_input_stats=True,
        )
        self.assertEqual(out.shape, [2, 3, 4, 4])

    def test_instance_norm_epsilon(self):
        """Test instance_norm with custom epsilon. / 测试自定义 epsilon 的 instance_norm。"""
        out = F.instance_norm(self.x, eps=1e-3, use_input_stats=True)
        self.assertEqual(out.shape, [2, 3, 4, 4])


class TestLocalResponseNorm(unittest.TestCase):
    """Tests for local_response_norm function. / local_response_norm 函数的测试。"""

    def test_local_response_norm_basic(self):
        """Test local_response_norm with basic params. / 测试基本参数的 local_response_norm。"""
        x = paddle.randn([2, 3, 4, 4], dtype='float32')
        out = F.local_response_norm(x, size=3)
        self.assertEqual(out.shape, [2, 3, 4, 4])

    def test_local_response_norm_with_params(self):
        """Test local_response_norm with custom params. / 测试自定义参数的 local_response_norm。"""
        x = paddle.randn([2, 3, 4, 4], dtype='float32')
        out = F.local_response_norm(x, size=5, alpha=0.001, beta=0.75, k=1.0)
        self.assertEqual(out.shape, [2, 3, 4, 4])


class TestGroupNorm(unittest.TestCase):
    """Tests for group_norm function. / group_norm 函数的测试。"""

    def setUp(self):
        self.x = paddle.randn([2, 8, 4, 4], dtype='float32')
        self.weight = paddle.ones([8], dtype='float32')
        self.bias = paddle.zeros([8], dtype='float32')

    def test_group_norm_basic(self):
        """Test group_norm with basic params. / 测试基本参数的 group_norm。"""
        out = F.group_norm(
            self.x, num_groups=4, weight=self.weight, bias=self.bias
        )
        self.assertEqual(out.shape, [2, 8, 4, 4])

    def test_group_norm_no_weight_bias(self):
        """Test group_norm without weight and bias. / 测试无权重偏置的 group_norm。"""
        out = F.group_norm(self.x, num_groups=4)
        self.assertEqual(out.shape, [2, 8, 4, 4])

    def test_group_norm_single_group(self):
        """Test group_norm with 1 group. / 测试单组的 group_norm。"""
        out = F.group_norm(self.x, num_groups=1)
        self.assertEqual(out.shape, [2, 8, 4, 4])

    def test_group_norm_channel_group(self):
        """Test group_norm with num_groups=channels. / 测试通道分组的 group_norm。"""
        out = F.group_norm(self.x, num_groups=8)
        self.assertEqual(out.shape, [2, 8, 4, 4])

    def test_group_norm_epsilon(self):
        """Test group_norm with custom epsilon. / 测试自定义 epsilon 的 group_norm。"""
        out = F.group_norm(self.x, num_groups=4, epsilon=1e-3)
        self.assertEqual(out.shape, [2, 8, 4, 4])

    def test_group_norm_nhwc(self):
        """Test group_norm with NHWC data format. / 测试 NHWC 格式的 group_norm。"""
        x_nhwc = self.x.transpose([0, 2, 3, 1])
        out = F.group_norm(x_nhwc, num_groups=4, data_format='NHWC')
        self.assertIsNotNone(out)

    def test_group_norm_invalid_data_format(self):
        """Test group_norm raises ValueError for invalid data_format. / 测试无效 data_format 时 group_norm 抛出 ValueError。"""
        with self.assertRaises(ValueError):
            F.group_norm(self.x, num_groups=4, data_format='invalid')


if __name__ == '__main__':
    unittest.main()
