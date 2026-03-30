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
Padding层单元测试 / Padding Layer Unit Tests

测试目标 / Test Target:
  paddle.nn Padding层 (覆盖率约82-84%)

覆盖的模块 / Covered Modules:
  - paddle.nn.Pad1D: 1D填充层
  - paddle.nn.Pad2D: 2D填充层
  - paddle.nn.Pad3D: 3D填充层
  - paddle.nn.functional.pad: 填充函数API

作用 / Purpose:
  覆盖各种填充模式（constant, reflect, replicate, circular）的代码路径。
"""

import unittest

import paddle
from paddle import nn

paddle.disable_static()


class TestPad1D(unittest.TestCase):
    """测试Pad1D填充层 / Test Pad1D padding layer"""

    def test_pad1d_constant(self):
        """测试常量填充 / Test constant padding"""
        pad = nn.Pad1D(padding=2, mode='constant', value=0.0)
        x = paddle.randn([4, 3, 10])
        y = pad(x)
        self.assertEqual(y.shape, [4, 3, 14])

    def test_pad1d_reflect(self):
        """测试反射填充 / Test reflect padding"""
        pad = nn.Pad1D(padding=1, mode='reflect')
        x = paddle.randn([4, 3, 10])
        y = pad(x)
        self.assertEqual(y.shape, [4, 3, 12])

    def test_pad1d_replicate(self):
        """测试复制填充 / Test replicate padding"""
        pad = nn.Pad1D(padding=2, mode='replicate')
        x = paddle.randn([4, 3, 10])
        y = pad(x)
        self.assertEqual(y.shape, [4, 3, 14])

    def test_pad1d_circular(self):
        """测试循环填充 / Test circular padding"""
        pad = nn.Pad1D(padding=3, mode='circular')
        x = paddle.randn([4, 3, 10])
        y = pad(x)
        self.assertEqual(y.shape, [4, 3, 16])

    def test_pad1d_asymmetric(self):
        """测试非对称填充 / Test asymmetric padding"""
        pad = nn.Pad1D(padding=[1, 3], mode='constant', value=0.0)
        x = paddle.randn([4, 3, 10])
        y = pad(x)
        self.assertEqual(y.shape, [4, 3, 14])  # 1 + 10 + 3

    def test_pad1d_nonzero_value(self):
        """测试非零常量填充 / Test non-zero constant padding"""
        pad = nn.Pad1D(padding=1, mode='constant', value=9.0)
        x = paddle.zeros([1, 1, 3])
        y = pad(x)
        self.assertAlmostEqual(float(y[0, 0, 0].numpy()), 9.0)
        self.assertAlmostEqual(float(y[0, 0, -1].numpy()), 9.0)


class TestPad2D(unittest.TestCase):
    """测试Pad2D填充层 / Test Pad2D padding layer"""

    def test_pad2d_constant(self):
        """测试2D常量填充 / Test 2D constant padding"""
        pad = nn.Pad2D(padding=2, mode='constant', value=0.0)
        x = paddle.randn([4, 3, 8, 8])
        y = pad(x)
        self.assertEqual(y.shape, [4, 3, 12, 12])

    def test_pad2d_reflect(self):
        """测试2D反射填充 / Test 2D reflect padding"""
        pad = nn.Pad2D(padding=1, mode='reflect')
        x = paddle.randn([4, 3, 8, 8])
        y = pad(x)
        self.assertEqual(y.shape, [4, 3, 10, 10])

    def test_pad2d_replicate(self):
        """测试2D复制填充 / Test 2D replicate padding"""
        pad = nn.Pad2D(padding=2, mode='replicate')
        x = paddle.randn([4, 3, 8, 8])
        y = pad(x)
        self.assertEqual(y.shape, [4, 3, 12, 12])

    def test_pad2d_circular(self):
        """测试2D循环填充 / Test 2D circular padding"""
        pad = nn.Pad2D(padding=2, mode='circular')
        x = paddle.randn([4, 3, 8, 8])
        y = pad(x)
        self.assertEqual(y.shape, [4, 3, 12, 12])

    def test_pad2d_asymmetric(self):
        """测试2D非对称填充 / Test 2D asymmetric padding"""
        pad = nn.Pad2D(padding=[1, 2, 3, 4], mode='constant', value=0.0)
        x = paddle.randn([4, 3, 8, 8])
        y = pad(x)
        # [left, right, top, bottom]
        self.assertEqual(y.shape, [4, 3, 8 + 3 + 4, 8 + 1 + 2])

    def test_pad2d_data_format(self):
        """测试不同数据格式 / Test different data format"""
        pad = nn.Pad2D(
            padding=1, mode='constant', value=0.0, data_format='NCHW'
        )
        x = paddle.randn([4, 3, 8, 8])
        y = pad(x)
        self.assertEqual(y.shape, [4, 3, 10, 10])

    def test_pad2d_value(self):
        """测试2D填充值 / Test 2D padding value"""
        pad = nn.Pad2D(padding=1, mode='constant', value=5.0)
        x = paddle.zeros([1, 1, 3, 3])
        y = pad(x)
        # Corner values should be 5.0
        self.assertAlmostEqual(float(y[0, 0, 0, 0].numpy()), 5.0)


class TestPad3D(unittest.TestCase):
    """测试Pad3D填充层 / Test Pad3D padding layer"""

    def test_pad3d_constant(self):
        """测试3D常量填充 / Test 3D constant padding"""
        pad = nn.Pad3D(padding=1, mode='constant', value=0.0)
        x = paddle.randn([2, 3, 4, 8, 8])
        y = pad(x)
        self.assertEqual(y.shape, [2, 3, 6, 10, 10])

    def test_pad3d_reflect(self):
        """测试3D反射填充 / Test 3D reflect padding"""
        pad = nn.Pad3D(padding=1, mode='reflect')
        x = paddle.randn([2, 3, 4, 8, 8])
        y = pad(x)
        self.assertEqual(y.shape, [2, 3, 6, 10, 10])

    def test_pad3d_replicate(self):
        """测试3D复制填充 / Test 3D replicate padding"""
        pad = nn.Pad3D(padding=1, mode='replicate')
        x = paddle.randn([2, 3, 4, 8, 8])
        y = pad(x)
        self.assertEqual(y.shape, [2, 3, 6, 10, 10])

    def test_pad3d_circular(self):
        """测试3D循环填充 / Test 3D circular padding"""
        pad = nn.Pad3D(padding=1, mode='circular')
        x = paddle.randn([2, 3, 4, 8, 8])
        y = pad(x)
        self.assertEqual(y.shape, [2, 3, 6, 10, 10])


class TestFunctionalPad(unittest.TestCase):
    """测试paddle.nn.functional.pad / Test paddle.nn.functional.pad"""

    def test_functional_pad_1d(self):
        """测试1D函数pad / Test 1D functional pad"""
        x = paddle.randn([4, 3, 10])
        y = paddle.nn.functional.pad(x, [2, 2], mode='constant', value=0.0)
        self.assertEqual(y.shape, [4, 3, 14])

    def test_functional_pad_2d(self):
        """测试2D函数pad / Test 2D functional pad"""
        x = paddle.randn([4, 3, 8, 8])
        y = paddle.nn.functional.pad(x, [1, 1, 1, 1], mode='reflect')
        self.assertEqual(y.shape, [4, 3, 10, 10])

    def test_functional_pad_3d(self):
        """测试3D函数pad / Test 3D functional pad"""
        x = paddle.randn([2, 3, 4, 8, 8])
        y = paddle.nn.functional.pad(x, [1, 1, 1, 1, 1, 1])
        self.assertEqual(y.shape, [2, 3, 6, 10, 10])

    def test_functional_pad_constant(self):
        """测试常量函数pad / Test constant functional pad"""
        x = paddle.zeros([1, 1, 3])
        y = paddle.nn.functional.pad(x, [2, 1], mode='constant', value=1.0)
        expected_size = 3 + 2 + 1
        self.assertEqual(y.shape[-1], expected_size)
        # Check padding values
        self.assertAlmostEqual(float(y[0, 0, 0].numpy()), 1.0)
        self.assertAlmostEqual(float(y[0, 0, -1].numpy()), 1.0)


if __name__ == '__main__':
    unittest.main()
