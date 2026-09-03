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
张量创建高级测试 / Advanced Tensor Creation Tests

测试目标 / Test Target:
  paddle.tensor.creation 高级创建函数

覆盖的模块 / Covered Modules:
  - paddle.linspace/logspace: 线性/对数等间隔序列
  - paddle.meshgrid: 网格生成
  - paddle.arange: 范围生成
  - paddle.eye: 单位矩阵
  - paddle.diag/diagflat/diagonal: 对角线操作
  - paddle.tril/triu: 三角矩阵

作用 / Purpose:
  补充张量创建API的测试，提升覆盖率。
"""

import unittest

import numpy as np

import paddle

paddle.disable_static()


class TestRangeOps(unittest.TestCase):
    """测试范围/序列生成 / Test range and sequence generation"""

    def test_arange_basic(self):
        """测试基本arange / Test basic arange"""
        result = paddle.arange(5)
        np.testing.assert_allclose(result.numpy(), [0, 1, 2, 3, 4])

    def test_arange_start_stop(self):
        """测试带起止的arange / Test arange with start and stop"""
        result = paddle.arange(2, 8, step=2)
        np.testing.assert_allclose(result.numpy(), [2, 4, 6])

    def test_arange_float(self):
        """测试浮点arange / Test float arange"""
        result = paddle.arange(0.0, 1.0, step=0.25)
        np.testing.assert_allclose(
            result.numpy(), [0.0, 0.25, 0.5, 0.75], rtol=1e-5
        )

    def test_linspace(self):
        """测试linspace / Test linspace"""
        result = paddle.linspace(0, 1, num=5)
        np.testing.assert_allclose(
            result.numpy(), [0.0, 0.25, 0.5, 0.75, 1.0], rtol=1e-5
        )

    def test_logspace(self):
        """测试logspace / Test logspace"""
        result = paddle.logspace(0, 2, num=3)  # 10^0, 10^1, 10^2
        np.testing.assert_allclose(
            result.numpy(), [1.0, 10.0, 100.0], rtol=1e-4
        )


class TestMeshGrid(unittest.TestCase):
    """测试网格生成 / Test meshgrid"""

    def test_meshgrid_2d(self):
        """测试2D网格 / Test 2D meshgrid"""
        x = paddle.to_tensor([1.0, 2.0, 3.0])
        y = paddle.to_tensor([4.0, 5.0])
        grid_x, grid_y = paddle.meshgrid(x, y)
        self.assertEqual(grid_x.shape, [3, 2])
        self.assertEqual(grid_y.shape, [3, 2])

    def test_meshgrid_3d(self):
        """测试3D网格 / Test 3D meshgrid"""
        x = paddle.arange(3)
        y = paddle.arange(4)
        z = paddle.arange(2)
        gx, gy, gz = paddle.meshgrid(x, y, z)
        self.assertEqual(gx.shape, [3, 4, 2])


class TestEyeAndDiag(unittest.TestCase):
    """测试eye和对角线操作 / Test eye and diagonal operations"""

    def test_eye_square(self):
        """测试方形单位矩阵 / Test square identity matrix"""
        result = paddle.eye(3)
        expected = np.eye(3)
        np.testing.assert_allclose(result.numpy(), expected)

    def test_eye_rectangular(self):
        """测试矩形单位矩阵 / Test rectangular identity matrix"""
        result = paddle.eye(3, 4)
        self.assertEqual(result.shape, [3, 4])
        # First 3 diagonal elements should be 1
        self.assertAlmostEqual(float(result[0, 0].numpy()), 1.0)

    def test_diag(self):
        """测试从向量创建对角矩阵 / Test diag from vector"""
        x = paddle.to_tensor([1.0, 2.0, 3.0])
        result = paddle.diag(x)
        self.assertEqual(result.shape, [3, 3])
        np.testing.assert_allclose(result.numpy().diagonal(), [1.0, 2.0, 3.0])

    def test_diag_extract(self):
        """测试从矩阵提取对角线 / Test diagonal extraction from matrix"""
        x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
        result = paddle.diag(x)
        np.testing.assert_allclose(result.numpy(), [1.0, 4.0])

    def test_diag_offset(self):
        """测试带偏移的对角线 / Test diagonal with offset"""
        x = paddle.to_tensor([1.0, 2.0, 3.0])
        result = paddle.diag(x, offset=1)
        self.assertEqual(result.shape, [4, 4])

    def test_diagonal(self):
        """测试diagonal操作 / Test diagonal operation"""
        x = paddle.to_tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        )
        result = paddle.diagonal(x)
        np.testing.assert_allclose(result.numpy(), [1.0, 5.0, 9.0])


class TestTriangularOps(unittest.TestCase):
    """测试三角矩阵操作 / Test triangular matrix operations"""

    def test_tril(self):
        """测试下三角 / Test lower triangular"""
        x = paddle.ones([3, 3])
        result = paddle.tril(x)
        expected = np.tril(np.ones([3, 3]))
        np.testing.assert_allclose(result.numpy(), expected)

    def test_triu(self):
        """测试上三角 / Test upper triangular"""
        x = paddle.ones([3, 3])
        result = paddle.triu(x)
        expected = np.triu(np.ones([3, 3]))
        np.testing.assert_allclose(result.numpy(), expected)

    def test_tril_offset(self):
        """测试带偏移的下三角 / Test lower triangular with offset"""
        x = paddle.ones([3, 3])
        result = paddle.tril(x, diagonal=1)
        self.assertEqual(result.shape, [3, 3])
        # Diagonal 1 means include one above main diagonal
        self.assertAlmostEqual(float(result[0, 1].numpy()), 1.0)
        self.assertAlmostEqual(float(result[0, 2].numpy()), 0.0)

    def test_triu_offset(self):
        """测试带偏移的上三角 / Test upper triangular with offset"""
        x = paddle.ones([3, 3])
        result = paddle.triu(x, diagonal=-1)
        self.assertEqual(result.shape, [3, 3])


class TestOnesZerosLike(unittest.TestCase):
    """测试ones/zeros_like函数 / Test ones/zeros_like functions"""

    def test_zeros_like(self):
        """测试zeros_like / Test zeros_like"""
        x = paddle.randn([2, 3, 4])
        result = paddle.zeros_like(x)
        self.assertEqual(result.shape, [2, 3, 4])
        self.assertAlmostEqual(float(result.sum().numpy()), 0.0)

    def test_ones_like(self):
        """测试ones_like / Test ones_like"""
        x = paddle.randn([2, 3, 4])
        result = paddle.ones_like(x)
        self.assertEqual(result.shape, [2, 3, 4])
        self.assertAlmostEqual(float(result.sum().numpy()), 24.0)

    def test_full(self):
        """测试full / Test full tensor creation"""
        result = paddle.full([3, 4], fill_value=3.14)
        self.assertEqual(result.shape, [3, 4])
        self.assertAlmostEqual(float(result[0, 0].numpy()), 3.14, places=4)

    def test_full_like(self):
        """测试full_like / Test full_like"""
        x = paddle.randn([2, 3])
        result = paddle.full_like(x, fill_value=5.0)
        self.assertEqual(result.shape, [2, 3])
        self.assertAlmostEqual(float(result[0, 0].numpy()), 5.0, places=4)


if __name__ == '__main__':
    unittest.main()
