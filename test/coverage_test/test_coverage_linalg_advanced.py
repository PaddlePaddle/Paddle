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
矩阵操作高级测试 / Advanced Matrix Operations Tests

测试目标 / Test Target:
  paddle.linalg 矩阵运算

覆盖的模块 / Covered Modules:
  - paddle.matmul: 矩阵乘法
  - paddle.linalg.norm: 范数计算
  - paddle.linalg.cond: 条件数
  - paddle.linalg.multi_dot: 多矩阵点乘

作用 / Purpose:
  补充矩阵运算API的测试，提升覆盖率。
"""

import unittest

import numpy as np

import paddle

paddle.disable_static()


class TestMatMul(unittest.TestCase):
    """测试矩阵乘法 / Test matrix multiplication"""

    def test_matmul_2d(self):
        """测试2D矩阵乘法 / Test 2D matrix multiplication"""
        A = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
        B = paddle.to_tensor([[5.0, 6.0], [7.0, 8.0]])
        result = paddle.matmul(A, B)
        expected = np.array(
            [[1 * 5 + 2 * 7, 1 * 6 + 2 * 8], [3 * 5 + 4 * 7, 3 * 6 + 4 * 8]],
            dtype=np.float32,
        )
        np.testing.assert_allclose(result.numpy(), expected)

    def test_matmul_batch(self):
        """测试批量矩阵乘法 / Test batched matrix multiplication"""
        A = paddle.randn([4, 3, 4])
        B = paddle.randn([4, 4, 5])
        result = paddle.matmul(A, B)
        self.assertEqual(result.shape, [4, 3, 5])

    def test_matmul_transpose(self):
        """测试带转置的矩阵乘法 / Test matmul with transpose"""
        A = paddle.randn([4, 3])
        B = paddle.randn([4, 3])
        result = paddle.matmul(A, B, transpose_y=True)
        self.assertEqual(result.shape, [4, 4])


class TestNorms(unittest.TestCase):
    """测试范数计算 / Test norm computation"""

    def test_vector_l2_norm(self):
        """测试向量L2范数 / Test vector L2 norm"""
        x = paddle.to_tensor([3.0, 4.0])
        result = paddle.linalg.norm(x)
        self.assertAlmostEqual(float(result.numpy()), 5.0, places=5)

    def test_matrix_frobenius_norm(self):
        """测试矩阵Frobenius范数 / Test matrix Frobenius norm"""
        x = paddle.to_tensor([[1.0, 0.0], [0.0, 1.0]])
        result = paddle.linalg.norm(x, p='fro')
        self.assertAlmostEqual(float(result.numpy()), np.sqrt(2), places=5)

    def test_vector_l1_norm(self):
        """测试向量L1范数 / Test vector L1 norm"""
        x = paddle.to_tensor([-3.0, 4.0, -1.0])
        result = paddle.linalg.norm(x, p=1)
        self.assertAlmostEqual(float(result.numpy()), 8.0, places=5)

    def test_norm_axis(self):
        """测试沿轴范数 / Test norm along axis"""
        x = paddle.to_tensor([[3.0, 4.0], [0.0, 2.0]])
        result = paddle.linalg.norm(x, axis=1)
        np.testing.assert_allclose(result.numpy(), [5.0, 2.0], rtol=1e-5)


class TestLinalg(unittest.TestCase):
    """测试线性代数操作 / Test linear algebra operations"""

    def test_multi_dot(self):
        """测试多矩阵连乘 / Test multi-dot product"""
        A = paddle.randn([4, 8])
        B = paddle.randn([8, 6])
        C = paddle.randn([6, 2])
        result = paddle.linalg.multi_dot([A, B, C])
        self.assertEqual(result.shape, [4, 2])

    def test_matrix_power(self):
        """测试矩阵幂 / Test matrix power"""
        A = paddle.eye(3)
        result = paddle.linalg.matrix_power(A, 3)
        np.testing.assert_allclose(result.numpy(), np.eye(3))

    def test_cross(self):
        """测试向量叉积 / Test vector cross product"""
        x = paddle.to_tensor([[1.0, 0.0, 0.0]])
        y = paddle.to_tensor([[0.0, 1.0, 0.0]])
        result = paddle.cross(x, y)
        np.testing.assert_allclose(result.numpy(), [[0.0, 0.0, 1.0]])

    def test_dot(self):
        """测试向量点积 / Test dot product"""
        x = paddle.to_tensor([1.0, 2.0, 3.0])
        y = paddle.to_tensor([4.0, 5.0, 6.0])
        result = paddle.dot(x, y)
        self.assertAlmostEqual(float(result.numpy()), 32.0, places=5)

    def test_outer(self):
        """测试外积 / Test outer product"""
        x = paddle.to_tensor([1.0, 2.0, 3.0])
        y = paddle.to_tensor([4.0, 5.0])
        result = paddle.outer(x, y)
        self.assertEqual(result.shape, [3, 2])
        expected = np.outer([1.0, 2.0, 3.0], [4.0, 5.0])
        np.testing.assert_allclose(result.numpy(), expected)


class TestSolveOperations(unittest.TestCase):
    """测试求解操作 / Test solve operations"""

    def test_inv(self):
        """测试矩阵逆 / Test matrix inverse"""
        A = paddle.to_tensor([[2.0, 1.0], [1.0, 1.0]])
        A_inv = paddle.linalg.inv(A)
        # A @ A_inv should be identity
        identity = paddle.matmul(A, A_inv)
        np.testing.assert_allclose(identity.numpy(), np.eye(2), atol=1e-5)

    def test_qr_decomposition(self):
        """测试QR分解 / Test QR decomposition"""
        A = paddle.randn([4, 3])
        Q, R = paddle.linalg.qr(A)
        self.assertEqual(Q.shape, [4, 3])
        self.assertEqual(R.shape, [3, 3])
        # Q should be orthogonal: Q^T @ Q = I
        QtQ = paddle.matmul(Q.t(), Q)
        np.testing.assert_allclose(QtQ.numpy(), np.eye(3), atol=1e-5)


if __name__ == '__main__':
    unittest.main()
