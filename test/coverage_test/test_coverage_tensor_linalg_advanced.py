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
进阶线性代数单元测试 / Advanced Linear Algebra Unit Tests

测试目标 / Test Target:
  paddle.tensor.linalg 进阶功能 (python/paddle/tensor/linalg.py, 覆盖率约68.2%)

覆盖的模块 / Covered Modules:
  - paddle.linalg.matrix_rank: 矩阵秩
  - paddle.linalg.cholesky: Cholesky分解
  - paddle.linalg.triangular_solve: 三角方程求解
  - paddle.linalg.lstsq: 最小二乘求解
  - paddle.linalg.pinv: 伪逆
  - paddle.linalg.eigvals: 特征值

作用 / Purpose:
  覆盖进阶线性代数操作的代码路径，补充矩阵分解和求解功能的测试。
"""

import unittest

import numpy as np

import paddle

paddle.disable_static()


class TestMatrixRankAndDet(unittest.TestCase):
    """测试矩阵秩和行列式 / Test matrix rank and determinant"""

    def test_matrix_rank_full(self):
        """测试满秩矩阵 / Test full rank matrix"""
        x = paddle.eye(3)
        rank = paddle.linalg.matrix_rank(x)
        self.assertEqual(int(rank.numpy()), 3)

    def test_matrix_rank_singular(self):
        """测试奇异矩阵 / Test singular matrix"""
        x = paddle.to_tensor([[1.0, 0.0], [2.0, 0.0]])
        rank = paddle.linalg.matrix_rank(x)
        self.assertEqual(int(rank.numpy()), 1)

    def test_det(self):
        """测试行列式 / Test determinant"""
        x = paddle.eye(3)
        det = paddle.linalg.det(x)
        self.assertAlmostEqual(float(det.numpy()), 1.0, places=5)

    def test_det_singular(self):
        """测试奇异矩阵行列式 / Test singular matrix determinant"""
        x = paddle.zeros([2, 2])
        det = paddle.linalg.det(x)
        self.assertAlmostEqual(float(det.numpy()), 0.0, places=5)

    def test_slogdet(self):
        """测试行列式对数 / Test log determinant"""
        x = paddle.eye(3) * 2
        sign, logabsdet = paddle.linalg.slogdet(x)
        self.assertIsNotNone(sign)
        self.assertIsNotNone(logabsdet)


class TestCholeskyDecomposition(unittest.TestCase):
    """测试Cholesky分解 / Test Cholesky decomposition"""

    def test_cholesky_basic(self):
        """测试基本Cholesky分解 / Test basic Cholesky decomposition"""
        # Create positive definite matrix
        A = paddle.to_tensor([[4.0, 2.0], [2.0, 3.0]])
        L = paddle.linalg.cholesky(A)
        self.assertEqual(L.shape, [2, 2])
        # Verify: L @ L^T ≈ A
        reconstructed = paddle.matmul(L, L.t())
        np.testing.assert_allclose(A.numpy(), reconstructed.numpy(), rtol=1e-5)

    def test_cholesky_upper(self):
        """测试上三角Cholesky / Test upper triangular Cholesky"""
        A = paddle.to_tensor([[4.0, 2.0], [2.0, 3.0]])
        U = paddle.linalg.cholesky(A, upper=True)
        self.assertEqual(U.shape, [2, 2])
        reconstructed = paddle.matmul(U.t(), U)
        np.testing.assert_allclose(A.numpy(), reconstructed.numpy(), rtol=1e-5)

    def test_cholesky_batch(self):
        """测试批量Cholesky分解 / Test batch Cholesky decomposition"""
        # Create batch of positive definite matrices
        A = paddle.to_tensor(
            [[[4.0, 2.0], [2.0, 3.0]], [[9.0, 3.0], [3.0, 2.0]]]
        )
        L = paddle.linalg.cholesky(A)
        self.assertEqual(L.shape, [2, 2, 2])


class TestSolveAndTriangular(unittest.TestCase):
    """测试方程求解 / Test equation solving"""

    def test_solve(self):
        """测试线性方程组求解 / Test linear system solving"""
        A = paddle.to_tensor([[2.0, 1.0], [1.0, 3.0]])
        b = paddle.to_tensor([[5.0], [10.0]])
        x = paddle.linalg.solve(A, b)
        # Verify Ax = b
        result = paddle.matmul(A, x)
        np.testing.assert_allclose(result.numpy(), b.numpy(), rtol=1e-4)

    def test_triangular_solve_lower(self):
        """测试下三角求解 / Test lower triangular solve"""
        A = paddle.to_tensor([[2.0, 0.0], [1.0, 3.0]])
        b = paddle.to_tensor([[2.0, 3.0], [5.0, 6.0]])  # b must be square
        x = paddle.linalg.triangular_solve(b, A, upper=False)
        self.assertEqual(x.shape, [2, 2])

    def test_triangular_solve_upper(self):
        """测试上三角求解 / Test upper triangular solve"""
        A = paddle.to_tensor([[2.0, 1.0], [0.0, 3.0]])
        b = paddle.to_tensor([[5.0, 1.0], [6.0, 2.0]])  # b must be square
        x = paddle.linalg.triangular_solve(b, A, upper=True)
        self.assertEqual(x.shape, [2, 2])


class TestPinvAndLstsq(unittest.TestCase):
    """测试伪逆和最小二乘 / Test pseudoinverse and least squares"""

    def test_pinv(self):
        """测试伪逆 / Test pseudoinverse"""
        A = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        pinv_A = paddle.linalg.pinv(A)
        self.assertEqual(pinv_A.shape, [2, 3])
        # Verify shape only (A @ pinv_A approximates identity only for square case)

    def test_lstsq(self):
        """测试最小二乘 / Test least squares"""
        A = paddle.to_tensor([[2.0, 1.0], [1.0, 3.0], [1.0, 1.0]])
        b = paddle.to_tensor([[5.0], [10.0], [4.0]])
        x, residuals, rank, sv = paddle.linalg.lstsq(A, b)
        self.assertEqual(x.shape, [2, 1])


class TestEigenDecomposition(unittest.TestCase):
    """测试特征值分解 / Test eigenvalue decomposition"""

    def test_eigvals(self):
        """测试特征值计算 / Test eigenvalue computation"""
        A = paddle.to_tensor([[1.0, 2.0], [2.0, 1.0]])
        eigvals = paddle.linalg.eigvals(A)
        self.assertEqual(eigvals.shape, [2])
        # Eigenvalues of [[1,2],[2,1]] are 3 and -1
        real_parts = paddle.real(eigvals).numpy()
        real_parts_sorted = np.sort(real_parts)
        np.testing.assert_allclose(
            real_parts_sorted, np.array([-1.0, 3.0]), rtol=1e-4
        )

    def test_eig(self):
        """测试特征值特征向量 / Test eigenvalue and eigenvectors"""
        A = paddle.to_tensor([[4.0, 0.0], [0.0, 3.0]])
        eigvals, eigvecs = paddle.linalg.eig(A)
        self.assertEqual(eigvals.shape, [2])
        self.assertEqual(eigvecs.shape, [2, 2])

    def test_eigh(self):
        """测试对称矩阵特征分解 / Test symmetric matrix eigendecomposition"""
        A = paddle.to_tensor([[2.0, 1.0], [1.0, 2.0]])
        eigvals, eigvecs = paddle.linalg.eigh(A)
        self.assertEqual(eigvals.shape, [2])
        # Verify A = V @ diag(eigenvalues) @ V^T
        reconstructed = paddle.matmul(
            paddle.matmul(eigvecs, paddle.diag(eigvals)), eigvecs.t()
        )
        np.testing.assert_allclose(A.numpy(), reconstructed.numpy(), atol=1e-5)

    def test_svd(self):
        """测试奇异值分解 / Test singular value decomposition"""
        A = paddle.randn([3, 4])
        U, S, Vh = paddle.linalg.svd(A, full_matrices=False)
        min_dim = min(3, 4)
        self.assertEqual(U.shape, [3, min_dim])
        self.assertEqual(S.shape, [min_dim])
        self.assertEqual(Vh.shape, [min_dim, 4])


if __name__ == '__main__':
    unittest.main()
