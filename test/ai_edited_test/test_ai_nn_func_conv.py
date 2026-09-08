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
# Target: paddle/nn/functional/conv.py
# Coverage target: improve coverage for conv functions (conv1d, conv2d, conv3d,
#   conv1d_transpose, conv2d_transpose, conv3d_transpose)
"""
Tests for paddle.nn.functional.conv module.
测试 paddle.nn.functional.conv 模块的单元测试。
"""

import unittest

import paddle
from paddle.nn import functional as F


class TestConv1D(unittest.TestCase):
    """Tests for conv1d function. / conv1d 函数的测试。"""

    def setUp(self):
        self.x = paddle.randn([2, 3, 10], dtype='float32')
        self.weight = paddle.randn([6, 3, 3], dtype='float32')

    def test_conv1d_basic(self):
        """Test conv1d with basic params. / 测试基本参数的 conv1d。"""
        out = F.conv1d(self.x, self.weight)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 6)

    def test_conv1d_with_stride(self):
        """Test conv1d with stride. / 测试带步幅的 conv1d。"""
        out = F.conv1d(self.x, self.weight, stride=2)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 6)

    def test_conv1d_with_padding(self):
        """Test conv1d with padding. / 测试带填充的 conv1d。"""
        out = F.conv1d(self.x, self.weight, padding=1)
        self.assertEqual(out.shape[2], 10)

    def test_conv1d_with_bias(self):
        """Test conv1d with bias. / 测试带偏置的 conv1d。"""
        bias = paddle.randn([6], dtype='float32')
        out = F.conv1d(self.x, self.weight, bias=bias)
        self.assertEqual(out.shape[1], 6)

    def test_conv1d_with_dilation(self):
        """Test conv1d with dilation. / 测试带膨胀的 conv1d。"""
        out = F.conv1d(self.x, self.weight, dilation=2)
        self.assertIsNotNone(out)

    def test_conv1d_with_groups(self):
        """Test conv1d with groups. / 测试带分组的 conv1d。"""
        weight_g = paddle.randn([6, 1, 3], dtype='float32')
        out = F.conv1d(self.x, weight_g, groups=3)
        self.assertEqual(out.shape[1], 6)

    def test_conv1d_data_format_nlc(self):
        """Test conv1d with NLC data format. / 测试 NLC 格式的 conv1d。"""
        x_nlc = self.x.transpose([0, 2, 1])  # [2, 10, 3]
        w_nlc = self.weight.reshape([6, 3, 3]).transpose([0, 2, 1])  # [6, 3, 3]
        out = F.conv1d(x_nlc, w_nlc, data_format='NLC')
        self.assertIsNotNone(out)

    def test_conv1d_same_padding(self):
        """Test conv1d with 'same' padding mode. / 测试 'same' 填充模式的 conv1d。"""
        out = F.conv1d(self.x, self.weight, padding='same')
        self.assertEqual(out.shape[2], 10)


class TestConv2D(unittest.TestCase):
    """Tests for conv2d function. / conv2d 函数的测试。"""

    def setUp(self):
        self.x = paddle.randn([2, 3, 8, 8], dtype='float32')
        self.weight = paddle.randn([6, 3, 3, 3], dtype='float32')

    def test_conv2d_basic(self):
        """Test conv2d with basic params. / 测试基本参数的 conv2d。"""
        out = F.conv2d(self.x, self.weight)
        self.assertEqual(out.shape, [2, 6, 6, 6])

    def test_conv2d_with_stride(self):
        """Test conv2d with stride. / 测试带步幅的 conv2d。"""
        out = F.conv2d(self.x, self.weight, stride=2)
        self.assertEqual(out.shape, [2, 6, 3, 3])

    def test_conv2d_with_padding(self):
        """Test conv2d with padding. / 测试带填充的 conv2d。"""
        out = F.conv2d(self.x, self.weight, padding=1)
        self.assertEqual(out.shape, [2, 6, 8, 8])

    def test_conv2d_with_bias(self):
        """Test conv2d with bias. / 测试带偏置的 conv2d。"""
        bias = paddle.randn([6], dtype='float32')
        out = F.conv2d(self.x, self.weight, bias=bias)
        self.assertEqual(out.shape, [2, 6, 6, 6])

    def test_conv2d_with_dilation(self):
        """Test conv2d with dilation. / 测试带膨胀的 conv2d。"""
        out = F.conv2d(self.x, self.weight, dilation=2)
        self.assertIsNotNone(out)

    def test_conv2d_with_groups(self):
        """Test conv2d with groups. / 测试带分组的 conv2d。"""
        weight_g = paddle.randn([6, 1, 3, 3], dtype='float32')
        out = F.conv2d(self.x, weight_g, groups=3)
        self.assertEqual(out.shape[1], 6)

    def test_conv2d_nhwc(self):
        """Test conv2d with NHWC data format. / 测试 NHWC 格式的 conv2d。"""
        x_nhwc = self.x.transpose([0, 2, 3, 1])  # [2, 8, 8, 3]
        w_nhwc = self.weight.transpose([0, 2, 3, 1])  # [6, 3, 3, 3]
        out = F.conv2d(x_nhwc, w_nhwc, data_format='NHWC')
        self.assertIsNotNone(out)

    def test_conv2d_same_padding(self):
        """Test conv2d with 'same' padding mode. / 测试 'same' 填充模式的 conv2d。"""
        out = F.conv2d(self.x, self.weight, padding='same')
        self.assertEqual(out.shape, [2, 6, 8, 8])

    def test_conv2d_asymmetric_padding(self):
        """Test conv2d with asymmetric padding. / 测试非对称填充的 conv2d。"""
        out = F.conv2d(self.x, self.weight, padding=[1, 2, 1, 2])
        self.assertIsNotNone(out)


class TestConv3D(unittest.TestCase):
    """Tests for conv3d function. / conv3d 函数的测试。"""

    def setUp(self):
        self.x = paddle.randn([2, 3, 4, 4, 4], dtype='float32')
        self.weight = paddle.randn([6, 3, 3, 3, 3], dtype='float32')

    def test_conv3d_basic(self):
        """Test conv3d with basic params. / 测试基本参数的 conv3d。"""
        out = F.conv3d(self.x, self.weight)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 6)

    def test_conv3d_with_stride(self):
        """Test conv3d with stride. / 测试带步幅的 conv3d。"""
        x = paddle.randn([2, 3, 6, 6, 6], dtype='float32')
        w = paddle.randn([6, 3, 3, 3, 3], dtype='float32')
        out = F.conv3d(x, w, stride=2)
        self.assertEqual(out.shape, [2, 6, 2, 2, 2])

    def test_conv3d_with_padding(self):
        """Test conv3d with padding. / 测试带填充的 conv3d。"""
        out = F.conv3d(self.x, self.weight, padding=1)
        self.assertEqual(out.shape, [2, 6, 4, 4, 4])

    def test_conv3d_with_bias(self):
        """Test conv3d with bias. / 测试带偏置的 conv3d。"""
        bias = paddle.randn([6], dtype='float32')
        out = F.conv3d(self.x, self.weight, bias=bias)
        self.assertEqual(out.shape[1], 6)

    def test_conv3d_ndhwc(self):
        """Test conv3d with NDHWC data format. / 测试 NDHWC 格式的 conv3d。"""
        x_ndhwc = self.x.transpose([0, 2, 3, 4, 1])
        w_ndhwc = self.weight.transpose([0, 2, 3, 4, 1])
        out = F.conv3d(x_ndhwc, w_ndhwc, data_format='NDHWC')
        self.assertIsNotNone(out)

    def test_conv3d_same_padding(self):
        """Test conv3d with 'same' padding mode. / 测试 'same' 填充模式的 conv3d。"""
        out = F.conv3d(self.x, self.weight, padding='same')
        self.assertEqual(out.shape, [2, 6, 4, 4, 4])


class TestConv1DTranspose(unittest.TestCase):
    """Tests for conv1d_transpose function. / conv1d_transpose 函数的测试。"""

    def setUp(self):
        self.x = paddle.randn([2, 6, 4], dtype='float32')
        self.weight = paddle.randn([6, 3, 3], dtype='float32')

    def test_conv1d_transpose_basic(self):
        """Test conv1d_transpose with basic params. / 测试基本参数的 conv1d_transpose。"""
        out = F.conv1d_transpose(self.x, self.weight)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 3)

    def test_conv1d_transpose_with_stride(self):
        """Test conv1d_transpose with stride. / 测试带步幅的 conv1d_transpose。"""
        out = F.conv1d_transpose(self.x, self.weight, stride=2)
        self.assertIsNotNone(out)

    def test_conv1d_transpose_with_padding(self):
        """Test conv1d_transpose with padding. / 测试带填充的 conv1d_transpose。"""
        out = F.conv1d_transpose(self.x, self.weight, padding=1)
        self.assertIsNotNone(out)

    def test_conv1d_transpose_with_bias(self):
        """Test conv1d_transpose with bias. / 测试带偏置的 conv1d_transpose。"""
        bias = paddle.randn([3], dtype='float32')
        out = F.conv1d_transpose(self.x, self.weight, bias=bias)
        self.assertIsNotNone(out)

    def test_conv1d_transpose_with_output_padding(self):
        """Test conv1d_transpose with output_padding. / 测试带 output_padding 的 conv1d_transpose。"""
        out = F.conv1d_transpose(
            self.x, self.weight, stride=2, output_padding=1
        )
        self.assertIsNotNone(out)

    def test_conv1d_transpose_dilation(self):
        """Test conv1d_transpose with dilation. / 测试带膨胀的 conv1d_transpose。"""
        out = F.conv1d_transpose(self.x, self.weight, dilation=2)
        self.assertIsNotNone(out)

    def test_conv1d_transpose_groups(self):
        """Test conv1d_transpose with groups. / 测试带分组的 conv1d_transpose。"""
        weight_g = paddle.randn([6, 1, 3], dtype='float32')
        out = F.conv1d_transpose(self.x, weight_g, groups=6)
        self.assertIsNotNone(out)


class TestConv2DTranspose(unittest.TestCase):
    """Tests for conv2d_transpose function. / conv2d_transpose 函数的测试。"""

    def setUp(self):
        self.x = paddle.randn([2, 6, 4, 4], dtype='float32')
        self.weight = paddle.randn([6, 3, 3, 3], dtype='float32')

    def test_conv2d_transpose_basic(self):
        """Test conv2d_transpose with basic params. / 测试基本参数的 conv2d_transpose。"""
        out = F.conv2d_transpose(self.x, self.weight)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 3)

    def test_conv2d_transpose_with_stride(self):
        """Test conv2d_transpose with stride. / 测试带步幅的 conv2d_transpose。"""
        out = F.conv2d_transpose(self.x, self.weight, stride=2)
        self.assertIsNotNone(out)

    def test_conv2d_transpose_with_padding(self):
        """Test conv2d_transpose with padding. / 测试带填充的 conv2d_transpose。"""
        out = F.conv2d_transpose(self.x, self.weight, padding=1)
        self.assertIsNotNone(out)

    def test_conv2d_transpose_with_bias(self):
        """Test conv2d_transpose with bias. / 测试带偏置的 conv2d_transpose。"""
        bias = paddle.randn([3], dtype='float32')
        out = F.conv2d_transpose(self.x, self.weight, bias=bias)
        self.assertIsNotNone(out)

    def test_conv2d_transpose_with_output_padding(self):
        """Test conv2d_transpose with output_padding. / 测试带 output_padding 的 conv2d_transpose。"""
        out = F.conv2d_transpose(
            self.x, self.weight, stride=2, output_padding=1
        )
        self.assertIsNotNone(out)

    def test_conv2d_transpose_dilation(self):
        """Test conv2d_transpose with dilation. / 测试带膨胀的 conv2d_transpose。"""
        out = F.conv2d_transpose(self.x, self.weight, dilation=2)
        self.assertIsNotNone(out)

    def test_conv2d_transpose_groups(self):
        """Test conv2d_transpose with groups. / 测试带分组的 conv2d_transpose。"""
        weight_g = paddle.randn([6, 1, 3, 3], dtype='float32')
        out = F.conv2d_transpose(self.x, weight_g, groups=6)
        self.assertIsNotNone(out)

    def test_conv2d_transpose_nhwc(self):
        """Test conv2d_transpose with NHWC format. / 测试 NHWC 格式的 conv2d_transpose。"""
        x_nhwc = self.x.transpose([0, 2, 3, 1])
        w_nhwc = self.weight.transpose([0, 2, 3, 1])
        out = F.conv2d_transpose(x_nhwc, w_nhwc, data_format='NHWC')
        self.assertIsNotNone(out)


class TestConv3DTranspose(unittest.TestCase):
    """Tests for conv3d_transpose function. / conv3d_transpose 函数的测试。"""

    def setUp(self):
        self.x = paddle.randn([2, 6, 2, 2, 2], dtype='float32')
        self.weight = paddle.randn([6, 3, 3, 3, 3], dtype='float32')

    def test_conv3d_transpose_basic(self):
        """Test conv3d_transpose with basic params. / 测试基本参数的 conv3d_transpose。"""
        out = F.conv3d_transpose(self.x, self.weight)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 3)

    def test_conv3d_transpose_with_stride(self):
        """Test conv3d_transpose with stride. / 测试带步幅的 conv3d_transpose。"""
        out = F.conv3d_transpose(self.x, self.weight, stride=2)
        self.assertIsNotNone(out)

    def test_conv3d_transpose_with_padding(self):
        """Test conv3d_transpose with padding. / 测试带填充的 conv3d_transpose。"""
        out = F.conv3d_transpose(self.x, self.weight, padding=1)
        self.assertIsNotNone(out)

    def test_conv3d_transpose_with_bias(self):
        """Test conv3d_transpose with bias. / 测试带偏置的 conv3d_transpose。"""
        bias = paddle.randn([3], dtype='float32')
        out = F.conv3d_transpose(self.x, self.weight, bias=bias)
        self.assertIsNotNone(out)

    def test_conv3d_transpose_ndhwc(self):
        """Test conv3d_transpose with NDHWC format. / 测试 NDHWC 格式的 conv3d_transpose。"""
        x_ndhwc = self.x.transpose([0, 2, 3, 4, 1])
        w_ndhwc = self.weight.transpose([0, 2, 3, 4, 1])
        out = F.conv3d_transpose(x_ndhwc, w_ndhwc, data_format='NDHWC')
        self.assertIsNotNone(out)


if __name__ == '__main__':
    unittest.main()
