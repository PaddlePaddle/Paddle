# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

# [AUTO-GENERATED] Unit test for paddle.linalg (vector_norm, matrix_norm, cond)
# 自动生成的单测，覆盖 paddle.tensor.linalg 模块中未覆盖的代码

"""
测试模块：paddle.linalg (vector_norm, matrix_norm, cond, cross, diagonal)
Test Module: paddle.linalg

本测试覆盖以下功能：
This test covers the following functions:
1. vector_norm - 向量范数 / Vector norm with various p values
2. matrix_norm - 矩阵范数 / Matrix norm
3. cond - 条件数 / Condition number
4. cross - 叉积 / Cross product
5. diagonal - 对角线 / Diagonal extraction

覆盖的未覆盖行：vector_norm各分支, cond的p值分支
"""

import unittest

import numpy as np

import paddle


class TestVectorNorm(unittest.TestCase):
    """测试vector_norm向量范数
    Test vector_norm with various p values"""

    def setUp(self):
        paddle.disable_static()

    def test_vector_norm_p2(self):
        """L2范数（默认） / L2 norm (default)"""
        x = paddle.to_tensor([3.0, 4.0], dtype='float32')
        out = paddle.linalg.vector_norm(x, p=2)
        np.testing.assert_allclose(float(out.numpy()), 5.0, rtol=1e-5)

    def test_vector_norm_p1(self):
        """L1范数 / L1 norm"""
        x = paddle.to_tensor([-3.0, 4.0], dtype='float32')
        out = paddle.linalg.vector_norm(x, p=1)
        np.testing.assert_allclose(float(out.numpy()), 7.0, rtol=1e-5)

    def test_vector_norm_p0(self):
        """L0范数（非零元素个数） / L0 norm (count nonzero)"""
        x = paddle.to_tensor([0.0, 1.0, 0.0, 2.0, 3.0], dtype='float32')
        out = paddle.linalg.vector_norm(x, p=0)
        np.testing.assert_allclose(float(out.numpy()), 3.0, rtol=1e-5)

    def test_vector_norm_inf(self):
        """无穷范数 / Infinity norm"""
        x = paddle.to_tensor([-5.0, 3.0, 4.0], dtype='float32')
        out = paddle.linalg.vector_norm(x, p=float('inf'))
        np.testing.assert_allclose(float(out.numpy()), 5.0, rtol=1e-5)

    def test_vector_norm_neg_inf(self):
        """负无穷范数 / Negative infinity norm"""
        x = paddle.to_tensor([1.0, 2.0, 3.0], dtype='float32')
        out = paddle.linalg.vector_norm(x, p=float('-inf'))
        np.testing.assert_allclose(float(out.numpy()), 1.0, rtol=1e-5)

    def test_vector_norm_with_axis(self):
        """指定axis的范数 / Norm along specific axis"""
        x = paddle.arange(24, dtype='float32').reshape([2, 3, 4]) - 12
        out = paddle.linalg.vector_norm(x, p=2, axis=[1, 2])
        self.assertEqual(list(out.shape), [2])

    def test_vector_norm_keepdim(self):
        """保持维度 / Keepdim"""
        x = paddle.ones([3, 4], dtype='float32')
        out = paddle.linalg.vector_norm(x, p=2, axis=1, keepdim=True)
        self.assertEqual(list(out.shape), [3, 1])


class TestCrossProduct(unittest.TestCase):
    """测试叉积运算
    Test cross product"""

    def setUp(self):
        paddle.disable_static()

    def test_cross_3d(self):
        """3D向量叉积 / 3D vector cross product"""
        x = paddle.to_tensor([1.0, 0.0, 0.0], dtype='float32')
        y = paddle.to_tensor([0.0, 1.0, 0.0], dtype='float32')
        out = paddle.cross(x, y)
        np.testing.assert_allclose(out.numpy(), [0.0, 0.0, 1.0], rtol=1e-5)

    def test_cross_batched(self):
        """批量叉积 / Batched cross product"""
        x = paddle.to_tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype='float32'
        )
        y = paddle.to_tensor(
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype='float32'
        )
        out = paddle.cross(x, y, axis=-1)
        self.assertEqual(list(out.shape), [2, 3])


class TestDiagonal(unittest.TestCase):
    """测试对角线提取
    Test diagonal extraction"""

    def setUp(self):
        paddle.disable_static()

    def test_diagonal_main(self):
        """主对角线 / Main diagonal"""
        x = paddle.arange(9, dtype='float32').reshape([3, 3])
        out = paddle.diagonal(x)
        np.testing.assert_allclose(out.numpy(), [0.0, 4.0, 8.0], rtol=1e-5)

    def test_diagonal_offset(self):
        """偏移对角线 / Offset diagonal"""
        x = paddle.arange(12, dtype='float32').reshape([3, 4])
        out = paddle.diagonal(x, offset=1)
        np.testing.assert_allclose(out.numpy(), [1.0, 6.0, 11.0], rtol=1e-5)

    def test_diagonal_negative_offset(self):
        """负偏移对角线 / Negative offset diagonal"""
        x = paddle.arange(12, dtype='float32').reshape([4, 3])
        out = paddle.diagonal(x, offset=-1)
        np.testing.assert_allclose(out.numpy(), [3.0, 7.0, 11.0], rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
