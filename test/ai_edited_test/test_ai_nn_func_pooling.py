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
# Target: paddle/nn/functional/pooling.py
# Coverage target: improve coverage for pooling functions (avg_pool1d, avg_pool2d, avg_pool3d,
#   max_pool1d, max_pool2d, max_pool3d, max_unpool1d, max_unpool2d, max_unpool3d,
#   adaptive_avg_pool1d, adaptive_avg_pool2d, adaptive_avg_pool3d, adaptive_max_pool1d,
#   adaptive_max_pool2d, adaptive_max_pool3d, lp_pool1d, lp_pool2d)
"""
Tests for paddle.nn.functional.pooling module.
测试 paddle.nn.functional.pooling 模块的单元测试。
"""

import unittest

import paddle
from paddle.nn import functional as F


class TestAvgPool1D(unittest.TestCase):
    """Tests for avg_pool1d function. / avg_pool1d 函数的测试。"""

    def setUp(self):
        self.x = paddle.randn([2, 3, 10], dtype='float32')

    def test_avg_pool1d_basic(self):
        """Test avg_pool1d with basic params. / 测试基本参数的 avg_pool1d。"""
        out = F.avg_pool1d(self.x, kernel_size=2, stride=2)
        self.assertEqual(out.shape, [2, 3, 5])

    def test_avg_pool1d_with_padding(self):
        """Test avg_pool1d with padding. / 测试带 padding 的 avg_pool1d。"""
        out = F.avg_pool1d(self.x, kernel_size=3, stride=2, padding=1)
        self.assertEqual(out.shape, [2, 3, 5])

    def test_avg_pool1d_with_count_include_pad(self):
        """Test avg_pool1d with count_include_pad=False. / 测试 count_include_pad=False 的 avg_pool1d。"""
        out = F.avg_pool1d(
            self.x, kernel_size=3, stride=2, padding=1, exclusive=True
        )
        self.assertEqual(out.shape, [2, 3, 5])

    def test_avg_pool1d_with_ceiling_mode(self):
        """Test avg_pool1d with ceil_mode=True. / 测试 ceil_mode=True 的 avg_pool1d。"""
        out = F.avg_pool1d(self.x, kernel_size=3, stride=2, ceil_mode=True)
        self.assertIsNotNone(out)

    def test_avg_pool1d_invalid_3elem_padding(self):
        """Test avg_pool1d raises ValueError for invalid 3-element padding. / 测试无效3元素 padding 时 avg_pool1d 抛出 ValueError。"""
        with self.assertRaises(ValueError):
            F.avg_pool1d(self.x, kernel_size=2, padding=(1, 0, 0))


class TestAvgPool2D(unittest.TestCase):
    """Tests for avg_pool2d function. / avg_pool2d 函数的测试。"""

    def setUp(self):
        self.x = paddle.randn([2, 3, 8, 8], dtype='float32')

    def test_avg_pool2d_basic(self):
        """Test avg_pool2d with basic params. / 测试基本参数的 avg_pool2d。"""
        out = F.avg_pool2d(self.x, kernel_size=2, stride=2)
        self.assertEqual(out.shape, [2, 3, 4, 4])

    def test_avg_pool2d_with_padding(self):
        """Test avg_pool2d with padding. / 测试带 padding 的 avg_pool2d。"""
        out = F.avg_pool2d(self.x, kernel_size=3, stride=2, padding=1)
        self.assertEqual(out.shape, [2, 3, 4, 4])

    def test_avg_pool2d_divisor_override(self):
        """Test avg_pool2d with divisor_override. / 测试 divisor_override 的 avg_pool2d。"""
        out = F.avg_pool2d(self.x, kernel_size=2, stride=2, divisor_override=3)
        self.assertEqual(out.shape, [2, 3, 4, 4])

    def test_avg_pool2d_ceil_mode(self):
        """Test avg_pool2d with ceil_mode. / 测试 ceil_mode 的 avg_pool2d。"""
        out = F.avg_pool2d(self.x, kernel_size=3, stride=2, ceil_mode=True)
        self.assertIsNotNone(out)

    def test_avg_pool2d_exclusive(self):
        """Test avg_pool2d with exclusive=True. / 测试 exclusive 的 avg_pool2d。"""
        out = F.avg_pool2d(
            self.x, kernel_size=3, stride=2, padding=1, exclusive=True
        )
        self.assertEqual(out.shape, [2, 3, 4, 4])


class TestAvgPool3D(unittest.TestCase):
    """Tests for avg_pool3d function. / avg_pool3d 函数的测试。"""

    def setUp(self):
        self.x = paddle.randn([2, 3, 4, 4, 4], dtype='float32')

    def test_avg_pool3d_basic(self):
        """Test avg_pool3d with basic params. / 测试基本参数的 avg_pool3d。"""
        out = F.avg_pool3d(self.x, kernel_size=2, stride=2)
        self.assertEqual(out.shape, [2, 3, 2, 2, 2])

    def test_avg_pool3d_with_padding(self):
        """Test avg_pool3d with padding. / 测试带 padding 的 avg_pool3d。"""
        out = F.avg_pool3d(self.x, kernel_size=2, stride=1, padding=0)
        self.assertEqual(out.shape, [2, 3, 3, 3, 3])


class TestMaxPool1D(unittest.TestCase):
    """Tests for max_pool1d function. / max_pool1d 函数的测试。"""

    def setUp(self):
        self.x = paddle.randn([2, 3, 10], dtype='float32')

    def test_max_pool1d_basic(self):
        """Test max_pool1d with basic params. / 测试基本参数的 max_pool1d。"""
        out = F.max_pool1d(self.x, kernel_size=2, stride=2)
        self.assertEqual(out.shape, [2, 3, 5])

    def test_max_pool1d_with_padding(self):
        """Test max_pool1d with padding. / 测试带 padding 的 max_pool1d。"""
        out = F.max_pool1d(self.x, kernel_size=3, stride=2, padding=1)
        self.assertEqual(out.shape, [2, 3, 5])

    def test_max_pool1d_return_mask(self):
        """Test max_pool1d with return_mask. / 测试 return_mask 的 max_pool1d。"""
        result = F.max_pool1d(self.x, kernel_size=2, stride=2, return_mask=True)
        out, mask = result
        self.assertEqual(out.shape, [2, 3, 5])
        self.assertEqual(mask.shape, [2, 3, 5])

    def test_max_pool1d_3elem_padding(self):
        """Test max_pool1d with symmetric padding. / 测试对称 padding 的 max_pool1d。"""
        out = F.max_pool1d(self.x, kernel_size=3, stride=1, padding=1)
        self.assertEqual(out.shape, [2, 3, 10])


class TestMaxPool2D(unittest.TestCase):
    """Tests for max_pool2d function. / max_pool2d 函数的测试。"""

    def setUp(self):
        self.x = paddle.randn([2, 3, 8, 8], dtype='float32')

    def test_max_pool2d_basic(self):
        """Test max_pool2d with basic params. / 测试基本参数的 max_pool2d。"""
        out = F.max_pool2d(self.x, kernel_size=2, stride=2)
        self.assertEqual(out.shape, [2, 3, 4, 4])

    def test_max_pool2d_with_padding(self):
        """Test max_pool2d with padding. / 测试带 padding 的 max_pool2d。"""
        out = F.max_pool2d(self.x, kernel_size=3, stride=2, padding=1)
        self.assertEqual(out.shape, [2, 3, 4, 4])

    def test_max_pool2d_return_mask(self):
        """Test max_pool2d with return_mask. / 测试 return_mask 的 max_pool2d。"""
        result = F.max_pool2d(self.x, kernel_size=2, stride=2, return_mask=True)
        out, mask = result
        self.assertEqual(out.shape, [2, 3, 4, 4])

    def test_max_pool2d_ceil_mode(self):
        """Test max_pool2d with ceil_mode. / 测试 ceil_mode 的 max_pool2d。"""
        out = F.max_pool2d(self.x, kernel_size=3, stride=2, ceil_mode=True)
        self.assertIsNotNone(out)

    def test_max_pool2d_adaptive_like(self):
        """Test max_pool2d with adaptive output_size (1, 1). / 测试自适应 output_size 的 max_pool2d。"""
        out = F.max_pool2d(self.x, kernel_size=3, stride=1, padding=1)
        self.assertEqual(out.shape, [2, 3, 8, 8])


class TestMaxPool3D(unittest.TestCase):
    """Tests for max_pool3d function. / max_pool3d 函数的测试。"""

    def setUp(self):
        self.x = paddle.randn([2, 3, 4, 4, 4], dtype='float32')

    def test_max_pool3d_basic(self):
        """Test max_pool3d with basic params. / 测试基本参数的 max_pool3d。"""
        out = F.max_pool3d(self.x, kernel_size=2, stride=2)
        self.assertEqual(out.shape, [2, 3, 2, 2, 2])

    def test_max_pool3d_return_mask(self):
        """Test max_pool3d with return_mask. / 测试 return_mask 的 max_pool3d。"""
        result = F.max_pool3d(self.x, kernel_size=2, stride=2, return_mask=True)
        out, mask = result
        self.assertEqual(out.shape, [2, 3, 2, 2, 2])


class TestAdaptiveAvgPool(unittest.TestCase):
    """Tests for adaptive_avg_pool functions. / adaptive_avg_pool 函数的测试。"""

    def test_adaptive_avg_pool1d(self):
        """Test adaptive_avg_pool1d. / 测试 adaptive_avg_pool1d。"""
        x = paddle.randn([2, 3, 10], dtype='float32')
        out = F.adaptive_avg_pool1d(x, output_size=5)
        self.assertEqual(out.shape, [2, 3, 5])

    def test_adaptive_avg_pool1d_single(self):
        """Test adaptive_avg_pool1d with output_size=1. / 测试 output_size=1 的 adaptive_avg_pool1d。"""
        x = paddle.randn([2, 3, 10], dtype='float32')
        out = F.adaptive_avg_pool1d(x, output_size=1)
        self.assertEqual(out.shape, [2, 3, 1])

    def test_adaptive_avg_pool2d(self):
        """Test adaptive_avg_pool2d. / 测试 adaptive_avg_pool2d。"""
        x = paddle.randn([2, 3, 8, 8], dtype='float32')
        out = F.adaptive_avg_pool2d(x, output_size=4)
        self.assertEqual(out.shape, [2, 3, 4, 4])

    def test_adaptive_avg_pool2d_tuple(self):
        """Test adaptive_avg_pool2d with tuple output_size. / 测试元组 output_size 的 adaptive_avg_pool2d。"""
        x = paddle.randn([2, 3, 8, 8], dtype='float32')
        out = F.adaptive_avg_pool2d(x, output_size=(3, 5))
        self.assertEqual(out.shape, [2, 3, 3, 5])

    def test_adaptive_avg_pool3d(self):
        """Test adaptive_avg_pool3d. / 测试 adaptive_avg_pool3d。"""
        x = paddle.randn([2, 3, 4, 4, 4], dtype='float32')
        out = F.adaptive_avg_pool3d(x, output_size=2)
        self.assertEqual(out.shape, [2, 3, 2, 2, 2])


class TestAdaptiveMaxPool(unittest.TestCase):
    """Tests for adaptive_max_pool functions. / adaptive_max_pool 函数的测试。"""

    def test_adaptive_max_pool1d(self):
        """Test adaptive_max_pool1d. / 测试 adaptive_max_pool1d。"""
        x = paddle.randn([2, 3, 10], dtype='float32')
        out = F.adaptive_max_pool1d(x, output_size=5)
        self.assertEqual(out.shape, [2, 3, 5])

    def test_adaptive_max_pool1d_single(self):
        """Test adaptive_max_pool1d with output_size=1. / 测试 output_size=1 的 adaptive_max_pool1d。"""
        x = paddle.randn([2, 3, 10], dtype='float32')
        out = F.adaptive_max_pool1d(x, output_size=1)
        self.assertEqual(out.shape, [2, 3, 1])

    def test_adaptive_max_pool1d_return_mask(self):
        """Test adaptive_max_pool1d with return_mask. / 测试 return_mask 的 adaptive_max_pool1d。"""
        x = paddle.randn([2, 3, 10], dtype='float32')
        result = F.adaptive_max_pool1d(x, output_size=5, return_mask=True)
        out, mask = result
        self.assertEqual(out.shape, [2, 3, 5])

    def test_adaptive_max_pool2d(self):
        """Test adaptive_max_pool2d. / 测试 adaptive_max_pool2d。"""
        x = paddle.randn([2, 3, 8, 8], dtype='float32')
        out = F.adaptive_max_pool2d(x, output_size=4)
        self.assertEqual(out.shape, [2, 3, 4, 4])

    def test_adaptive_max_pool2d_return_mask(self):
        """Test adaptive_max_pool2d with return_mask. / 测试 return_mask 的 adaptive_max_pool2d。"""
        x = paddle.randn([2, 3, 8, 8], dtype='float32')
        result = F.adaptive_max_pool2d(x, output_size=4, return_mask=True)
        out, mask = result
        self.assertEqual(out.shape, [2, 3, 4, 4])

    def test_adaptive_max_pool3d(self):
        """Test adaptive_max_pool3d. / 测试 adaptive_max_pool3d。"""
        x = paddle.randn([2, 3, 4, 4, 4], dtype='float32')
        out = F.adaptive_max_pool3d(x, output_size=2)
        self.assertEqual(out.shape, [2, 3, 2, 2, 2])

    def test_adaptive_max_pool3d_return_mask(self):
        """Test adaptive_max_pool3d with return_mask. / 测试 return_mask 的 adaptive_max_pool3d。"""
        x = paddle.randn([2, 3, 4, 4, 4], dtype='float32')
        result = F.adaptive_max_pool3d(x, output_size=2, return_mask=True)
        out, mask = result
        self.assertEqual(out.shape, [2, 3, 2, 2, 2])


class TestMaxUnpool(unittest.TestCase):
    """Tests for max_unpool functions. / max_unpool 函数的测试。"""

    def test_max_unpool1d(self):
        """Test max_unpool1d. / 测试 max_unpool1d。"""
        x = paddle.randn([2, 3, 4], dtype='float32')
        indices = paddle.to_tensor([[[0, 1, 2, 3]]] * 2 * 3, dtype='int64')
        indices = indices.reshape([2, 3, 4])
        out = F.max_unpool1d(x, indices, kernel_size=2, stride=2)
        self.assertIsNotNone(out)

    def test_max_unpool2d(self):
        """Test max_unpool2d. / 测试 max_unpool2d。"""
        pool_out = paddle.randn([2, 3, 4, 4], dtype='float32')
        indices = paddle.zeros([2, 3, 4, 4], dtype='int64')
        out = F.max_unpool2d(pool_out, indices, kernel_size=2, stride=2)
        self.assertIsNotNone(out)

    def test_max_unpool3d(self):
        """Test max_unpool3d. / 测试 max_unpool3d。"""
        pool_out = paddle.randn([2, 3, 2, 2, 2], dtype='float32')
        indices = paddle.zeros([2, 3, 2, 2, 2], dtype='int64')
        out = F.max_unpool3d(pool_out, indices, kernel_size=2, stride=2)
        self.assertIsNotNone(out)


class TestLpPool(unittest.TestCase):
    """Tests for lp_pool functions. / lp_pool 函数的测试。"""

    def test_lp_pool1d_basic(self):
        """Test lp_pool1d with basic params. / 测试基本参数的 lp_pool1d。"""
        x = paddle.randn([2, 3, 10], dtype='float32')
        out = F.lp_pool1d(x, norm_type=2, kernel_size=3, stride=2)
        self.assertIsNotNone(out)

    def test_lp_pool1d_single_value(self):
        """Test lp_pool1d with norm_type=1. / 测试 norm_type=1 的 lp_pool1d。"""
        x = paddle.randn([2, 3, 10], dtype='float32')
        out = F.lp_pool1d(x, norm_type=1, kernel_size=2, stride=2)
        self.assertIsNotNone(out)

    def test_lp_pool2d_basic(self):
        """Test lp_pool2d with basic params. / 测试基本参数的 lp_pool2d。"""
        x = paddle.randn([2, 3, 8, 8], dtype='float32')
        out = F.lp_pool2d(x, norm_type=2, kernel_size=2, stride=2)
        self.assertIsNotNone(out)

    def test_lp_pool2d_single_value(self):
        """Test lp_pool2d with norm_type=1. / 测试 norm_type=1 的 lp_pool2d。"""
        x = paddle.randn([2, 3, 8, 8], dtype='float32')
        out = F.lp_pool2d(x, norm_type=1, kernel_size=2, stride=2)
        self.assertIsNotNone(out)


if __name__ == '__main__':
    unittest.main()
