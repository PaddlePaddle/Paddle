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

# [AUTO-GENERATED] Test file for paddle.tensor.linalg
# 覆盖模块: paddle/tensor/linalg.py
# Uncovered lines: transpose, matmul, bmm, dot, mv, cross, dist, cholesky,
#   cholesky_inverse, cholesky_solve, lu, lu_solve, lu_unpack, qr, svd,
#   eigh, eigvals, eigvalsh, solve, inv, pinv, cond, det, slogdet,
#   matrix_power, multi_dot, matrix_norm, vector_norm, norm,
#   matrix_transpose, householder_product, cdist, corrcoef, cov,
#   histogram, histogram_bin_edges, vecdot, matrix_exp, lstsq,
#   svdvals, svd_lowrank, pca_lowrank, triangular_solve

import unittest

import numpy as np

import paddle


class TestTranspose(unittest.TestCase):
    """测试 transpose 函数
    Test transpose function"""

    def test_transpose_2d(self):
        """测试二维张量转置
        Test 2D tensor transpose"""
        x = paddle.randn([3, 4])
        result = paddle.transpose(x, [1, 0])
        self.assertEqual(result.shape, [4, 3])

    def test_transpose_3d(self):
        """测试三维张量转置
        Test 3D tensor transpose"""
        x = paddle.randn([2, 3, 4])
        result = paddle.transpose(x, [2, 0, 1])
        self.assertEqual(result.shape, [4, 2, 3])

    def test_transpose_roundtrip(self):
        """测试转置后还原
        Test transpose roundtrip"""
        x = paddle.randn([2, 3, 4])
        result = paddle.transpose(paddle.transpose(x, [2, 0, 1]), [1, 2, 0])
        np.testing.assert_allclose(result.numpy(), x.numpy(), atol=1e-6)


class TestMatmul(unittest.TestCase):
    """测试 matmul 函数
    Test matmul function"""

    def test_matmul_2d(self):
        """测试二维矩阵乘法
        Test 2D matrix multiplication"""
        a = paddle.randn([3, 4])
        b = paddle.randn([4, 5])
        result = paddle.matmul(a, b)
        self.assertEqual(result.shape, [3, 5])

    def test_matmul_1d(self):
        """测试向量点积
        Test vector dot product via matmul"""
        a = paddle.randn([4])
        b = paddle.randn([4])
        result = paddle.matmul(a, b)
        self.assertEqual(result.shape, [])

    def test_matmul_batched(self):
        """测试批量矩阵乘法
        Test batched matrix multiplication"""
        a = paddle.randn([2, 3, 4])
        b = paddle.randn([2, 4, 5])
        result = paddle.matmul(a, b)
        self.assertEqual(result.shape, [2, 3, 5])

    def test_matmul_broadcast(self):
        """测试广播矩阵乘法
        Test broadcast matrix multiplication"""
        a = paddle.randn([2, 3, 4])
        b = paddle.randn([4, 5])
        result = paddle.matmul(a, b)
        self.assertEqual(result.shape, [2, 3, 5])


class TestBmm(unittest.TestCase):
    """测试 bmm 函数
    Test bmm function"""

    def test_bmm_basic(self):
        """测试基本批量矩阵乘法
        Test basic batched matrix multiplication"""
        a = paddle.randn([2, 3, 4])
        b = paddle.randn([2, 4, 5])
        result = paddle.bmm(a, b)
        self.assertEqual(result.shape, [2, 3, 5])

    def test_bmm_correctness(self):
        """测试 bmm 结果正确性
        Test bmm correctness"""
        a_np = np.random.randn(2, 3, 4).astype('float32')
        b_np = np.random.randn(2, 4, 5).astype('float32')
        a = paddle.to_tensor(a_np)
        b = paddle.to_tensor(b_np)
        result = paddle.bmm(a, b)
        expected = np.matmul(a_np, b_np)
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-5)


class TestDot(unittest.TestCase):
    """测试 dot 函数
    Test dot function"""

    def test_dot_1d(self):
        """测试一维向量点积
        Test 1D vector dot product"""
        a = paddle.to_tensor([1.0, 2.0, 3.0])
        b = paddle.to_tensor([4.0, 5.0, 6.0])
        result = paddle.dot(a, b)
        self.assertAlmostEqual(result.item(), 32.0, places=5)

    def test_dot_correctness(self):
        """测试 dot 结果正确性
        Test dot correctness"""
        a = paddle.randn([5])
        b = paddle.randn([5])
        result = paddle.dot(a, b)
        expected = np.dot(a.numpy(), b.numpy())
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-5)


class TestMv(unittest.TestCase):
    """测试 mv 函数
    Test mv function"""

    def test_mv_basic(self):
        """测试矩阵向量乘法
        Test matrix-vector multiplication"""
        a = paddle.randn([3, 4])
        b = paddle.randn([4])
        result = paddle.mv(a, b)
        self.assertEqual(result.shape, [3])

    def test_mv_correctness(self):
        """测试 mv 结果正确性
        Test mv correctness"""
        a_np = np.random.randn(3, 4).astype('float32')
        b_np = np.random.randn(4).astype('float32')
        result = paddle.mv(paddle.to_tensor(a_np), paddle.to_tensor(b_np))
        expected = np.dot(a_np, b_np)
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-5)


class TestCross(unittest.TestCase):
    """测试 cross 函数
    Test cross function"""

    def test_cross_3d(self):
        """测试三维向量叉积
        Test 3D vector cross product"""
        a = paddle.to_tensor([1.0, 0.0, 0.0])
        b = paddle.to_tensor([0.0, 1.0, 0.0])
        result = paddle.cross(a, b)
        expected = np.array([0.0, 0.0, 1.0])
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-5)

    def test_cross_batch(self):
        """测试批量叉积
        Test batched cross product"""
        a = paddle.randn([2, 3])
        b = paddle.randn([2, 3])
        result = paddle.cross(a, b)
        self.assertEqual(result.shape, [2, 3])


class TestDist(unittest.TestCase):
    """测试 dist 函数
    Test dist function"""

    def test_dist_basic(self):
        """测试基本距离计算
        Test basic distance computation"""
        a = paddle.to_tensor([1.0, 2.0, 3.0])
        b = paddle.to_tensor([1.0, 2.0, 3.0])
        result = paddle.dist(a, b)
        self.assertAlmostEqual(result.item(), 0.0, places=5)

    def test_dist_nonzero(self):
        """测试非零距离计算
        Test non-zero distance computation"""
        a = paddle.to_tensor([0.0, 0.0])
        b = paddle.to_tensor([3.0, 4.0])
        result = paddle.dist(a, b)
        self.assertAlmostEqual(result.item(), 5.0, places=4)


class TestCholesky(unittest.TestCase):
    """测试 cholesky 分解
    Test cholesky decomposition"""

    def test_cholesky_basic(self):
        """测试基本 cholesky 分解
        Test basic cholesky decomposition"""
        a = paddle.randn([4, 4])
        a = paddle.matmul(a, a.T) + paddle.eye(4) * 0.5
        L = paddle.linalg.cholesky(a)
        self.assertEqual(L.shape, [4, 4])
        # Verify L * L^T ≈ a
        reconstructed = paddle.matmul(L, L.T)
        np.testing.assert_allclose(reconstructed.numpy(), a.numpy(), atol=1e-4)

    def test_cholesky_batched(self):
        """测试批量 cholesky 分解
        Test batched cholesky decomposition"""
        a = paddle.randn([2, 4, 4])
        a = paddle.matmul(a, a.transpose([0, 2, 1])) + paddle.eye(4) * 0.5
        L = paddle.linalg.cholesky(a)
        self.assertEqual(L.shape, [2, 4, 4])

    def test_cholesky_upper(self):
        """测试上三角 cholesky 分解
        Test upper triangular cholesky"""
        a = paddle.randn([3, 3])
        a = paddle.matmul(a, a.T) + paddle.eye(3) * 0.5
        U = paddle.linalg.cholesky(a, upper=True)
        self.assertEqual(U.shape, [3, 3])


class TestCholeskyInverse(unittest.TestCase):
    """测试 cholesky_inverse 函数
    Test cholesky_inverse function"""

    def test_cholesky_inverse_basic(self):
        """测试基本 cholesky 逆
        Test basic cholesky inverse"""
        a = paddle.randn([3, 3])
        a = paddle.matmul(a, a.T) + paddle.eye(3) * 0.5
        L = paddle.linalg.cholesky(a)
        inv = paddle.linalg.cholesky_inverse(L)
        self.assertEqual(inv.shape, [3, 3])
        # Verify A * inv(A) ≈ I
        result = paddle.matmul(a, inv)
        np.testing.assert_allclose(result.numpy(), np.eye(3), atol=1e-4)


class TestCholeskySolve(unittest.TestCase):
    """测试 cholesky_solve 函数
    Test cholesky_solve function"""

    def test_cholesky_solve_basic(self):
        """测试基本 cholesky 求解
        Test basic cholesky solve"""
        a = paddle.randn([3, 3])
        a = paddle.matmul(a, a.T) + paddle.eye(3) * 0.5
        L = paddle.linalg.cholesky(a)
        b = paddle.randn([3, 1])
        # cholesky_solve(x, y): y is the Cholesky factor, x is the RHS
        x = paddle.linalg.cholesky_solve(b, L)
        self.assertEqual(x.shape, [3, 1])


class TestLU(unittest.TestCase):
    """测试 LU 分解
    Test LU decomposition"""

    def test_lu_basic(self):
        """测试基本 LU 分解
        Test basic LU decomposition"""
        a = paddle.randn([3, 3])
        lu, p = paddle.linalg.lu(a)
        self.assertEqual(lu.shape, [3, 3])
        self.assertEqual(p.shape, [3])

    def test_lu_with_infos(self):
        """测试带 info 的 LU 分解
        Test LU decomposition with info"""
        a = paddle.randn([3, 3])
        lu, p, info = paddle.linalg.lu(a, get_infos=True)
        self.assertEqual(lu.shape, [3, 3])
        self.assertEqual(p.shape, [3])
        self.assertEqual(info.shape, [])

    def test_lu_unpack(self):
        """测试 LU 解包
        Test LU unpack"""
        a = paddle.randn([3, 3])
        lu, p = paddle.linalg.lu(a)
        P, L, U = paddle.linalg.lu_unpack(lu, p)
        self.assertEqual(P.shape, [3, 3])
        self.assertEqual(L.shape, [3, 3])
        self.assertEqual(U.shape, [3, 3])

    def test_lu_solve(self):
        """测试 LU 求解
        Test LU solve"""
        a = paddle.randn([3, 3])
        b = paddle.randn([3, 1])
        lu, p = paddle.linalg.lu(a)
        x = paddle.linalg.lu_solve(b, lu, p)
        self.assertEqual(x.shape, [3, 1])


class TestQR(unittest.TestCase):
    """测试 QR 分解
    Test QR decomposition"""

    def test_qr_basic(self):
        """测试基本 QR 分解
        Test basic QR decomposition"""
        a = paddle.randn([4, 3])
        q, r = paddle.linalg.qr(a)
        self.assertEqual(q.shape, [4, 3])
        self.assertEqual(r.shape, [3, 3])

    def test_qr_reduced(self):
        """测试缩减 QR 分解
        Test reduced QR decomposition"""
        a = paddle.randn([5, 3])
        q, r = paddle.linalg.qr(a, mode='reduced')
        self.assertEqual(q.shape, [5, 3])
        self.assertEqual(r.shape, [3, 3])

    def test_qr_complete(self):
        """测试完全 QR 分解
        Test complete QR decomposition"""
        a = paddle.randn([5, 3])
        q, r = paddle.linalg.qr(a, mode='complete')
        self.assertEqual(q.shape, [5, 5])
        self.assertEqual(r.shape, [5, 3])


class TestSVD(unittest.TestCase):
    """测试 SVD 分解
    Test SVD decomposition"""

    def test_svd_basic(self):
        """测试基本 SVD 分解
        Test basic SVD decomposition"""
        a = paddle.randn([4, 3])
        u, s, vh = paddle.linalg.svd(a)
        self.assertEqual(u.shape, [4, 3])
        self.assertEqual(s.shape, [3])
        self.assertEqual(vh.shape, [3, 3])

    def test_svd_reconstruction(self):
        """测试 SVD 重构
        Test SVD reconstruction"""
        a = paddle.randn([4, 3])
        u, s, vh = paddle.linalg.svd(a)
        # Reconstruct: U @ diag(s) @ Vh
        reconstructed = paddle.matmul(u[:, :3] * s.unsqueeze(0), vh)
        np.testing.assert_allclose(reconstructed.numpy(), a.numpy(), atol=1e-4)


class TestEigh(unittest.TestCase):
    """测试 eigh 特征值分解
    Test eigh eigenvalue decomposition"""

    def test_eigh_basic(self):
        """测试基本对称特征值分解
        Test basic symmetric eigenvalue decomposition"""
        a = paddle.randn([3, 3])
        a = (a + a.T) / 2
        eigenvalues, eigenvectors = paddle.linalg.eigh(a)
        self.assertEqual(eigenvalues.shape, [3])
        self.assertEqual(eigenvectors.shape, [3, 3])

    def test_eigh_ascending(self):
        """测试升序特征值
        Test ascending eigenvalues"""
        a = paddle.randn([3, 3])
        a = (a + a.T) / 2
        eigenvalues, _ = paddle.linalg.eigh(a)
        # Default is ascending
        diffs = eigenvalues[1:] - eigenvalues[:-1]
        self.assertTrue(paddle.all(diffs >= -1e-5).item())


class TestEigvalsh(unittest.TestCase):
    """测试 eigvalsh 函数
    Test eigvalsh function"""

    def test_eigvalsh_basic(self):
        """测试基本特征值计算
        Test basic eigenvalue computation"""
        a = paddle.randn([3, 3])
        a = (a + a.T) / 2
        eigenvalues = paddle.linalg.eigvalsh(a)
        self.assertEqual(eigenvalues.shape, [3])


class TestSolve(unittest.TestCase):
    """测试 solve 函数
    Test solve function"""

    def test_solve_basic(self):
        """测试基本线性方程组求解
        Test basic linear system solve"""
        a = paddle.randn([3, 3])
        b = paddle.randn([3, 1])
        x = paddle.linalg.solve(a, b)
        self.assertEqual(x.shape, [3, 1])

    def test_solve_correctness(self):
        """测试求解正确性
        Test solve correctness"""
        a_np = np.random.randn(3, 3).astype('float32')
        b_np = np.random.randn(3, 1).astype('float32')
        x = paddle.linalg.solve(paddle.to_tensor(a_np), paddle.to_tensor(b_np))
        expected = np.linalg.solve(a_np, b_np)
        np.testing.assert_allclose(x.numpy(), expected, atol=1e-4)


class TestInv(unittest.TestCase):
    """测试 inv 函数
    Test inv function"""

    def test_inv_basic(self):
        """测试基本矩阵求逆
        Test basic matrix inversion"""
        a = paddle.randn([3, 3])
        a = a + paddle.eye(3) * 5  # Make well-conditioned
        inv_a = paddle.linalg.inv(a)
        self.assertEqual(inv_a.shape, [3, 3])

    def test_inv_correctness(self):
        """测试逆矩阵正确性
        Test inverse matrix correctness"""
        a_np = np.random.randn(3, 3).astype('float32')
        a_np = a_np + np.eye(3) * 5
        a = paddle.to_tensor(a_np)
        inv_a = paddle.linalg.inv(a)
        result = paddle.matmul(a, inv_a)
        np.testing.assert_allclose(result.numpy(), np.eye(3), atol=1e-4)


class TestPinv(unittest.TestCase):
    """测试 pinv 函数
    Test pinv function"""

    def test_pinv_basic(self):
        """测试基本伪逆计算
        Test basic pseudo-inverse computation"""
        a = paddle.randn([3, 4])
        pinv_a = paddle.linalg.pinv(a)
        self.assertEqual(pinv_a.shape, [4, 3])

    def test_pinv_square(self):
        """测试方阵伪逆
        Test square matrix pseudo-inverse"""
        a = paddle.randn([3, 3])
        a = a + paddle.eye(3) * 5
        pinv_a = paddle.linalg.pinv(a)
        self.assertEqual(pinv_a.shape, [3, 3])


class TestCond(unittest.TestCase):
    """测试 cond 函数
    Test cond function"""

    def test_cond_basic(self):
        """测试基本条件数计算
        Test basic condition number computation"""
        a = paddle.randn([3, 3])
        result = paddle.linalg.cond(a)
        self.assertEqual(result.shape, [])
        self.assertGreater(result.item(), 0)

    def test_cond_identity(self):
        """测试单位矩阵条件数为1
        Test identity matrix condition number is 1"""
        a = paddle.eye(3)
        result = paddle.linalg.cond(a)
        self.assertAlmostEqual(result.item(), 1.0, places=4)


class TestDet(unittest.TestCase):
    """测试 det 函数
    Test det function"""

    def test_det_basic(self):
        """测试基本行列式计算
        Test basic determinant computation"""
        a = paddle.randn([3, 3])
        result = paddle.linalg.det(a)
        self.assertEqual(result.shape, [])

    def test_det_identity(self):
        """测试单位矩阵行列式为1
        Test identity matrix determinant is 1"""
        a = paddle.eye(3)
        result = paddle.linalg.det(a)
        self.assertAlmostEqual(result.item(), 1.0, places=5)


class TestSlogdet(unittest.TestCase):
    """测试 slogdet 函数
    Test slogdet function"""

    def test_slogdet_basic(self):
        """测试基本符号对数行列式
        Test basic sign-log-determinant"""
        a = paddle.randn([3, 3])
        sign, logabsdet = paddle.linalg.slogdet(a)
        self.assertEqual(sign.shape, [])
        self.assertEqual(logabsdet.shape, [])

    def test_slogdet_positive(self):
        """测试正定矩阵的 slogdet
        Test slogdet of positive definite matrix"""
        a = paddle.randn([3, 3])
        a = paddle.matmul(a, a.T) + paddle.eye(3)
        sign, logabsdet = paddle.linalg.slogdet(a)
        self.assertAlmostEqual(sign.item(), 1.0, places=5)


class TestMatrixPower(unittest.TestCase):
    """测试 matrix_power 函数
    Test matrix_power function"""

    def test_matrix_power_square(self):
        """测试矩阵平方
        Test matrix square"""
        a = paddle.randn([3, 3])
        result = paddle.linalg.matrix_power(a, 2)
        self.assertEqual(result.shape, [3, 3])

    def test_matrix_power_identity(self):
        """测试矩阵零次幂为单位矩阵
        Test matrix power 0 is identity"""
        a = paddle.randn([3, 3])
        result = paddle.linalg.matrix_power(a, 0)
        np.testing.assert_allclose(result.numpy(), np.eye(3), atol=1e-5)

    def test_matrix_power_neg(self):
        """测试矩阵负幂为逆
        Test negative matrix power"""
        a = paddle.randn([3, 3])
        a = a + paddle.eye(3) * 5
        result = paddle.linalg.matrix_power(a, -1)
        expected = paddle.linalg.inv(a)
        np.testing.assert_allclose(result.numpy(), expected.numpy(), atol=1e-4)


class TestMultiDot(unittest.TestCase):
    """测试 multi_dot 函数
    Test multi_dot function"""

    def test_multi_dot_3(self):
        """测试三个矩阵连乘
        Test 3-matrix chain multiplication"""
        a = paddle.randn([2, 3])
        b = paddle.randn([3, 4])
        c = paddle.randn([4, 2])
        result = paddle.linalg.multi_dot([a, b, c])
        self.assertEqual(result.shape, [2, 2])

    def test_multi_dot_2(self):
        """测试两个矩阵连乘
        Test 2-matrix chain multiplication"""
        a = paddle.randn([3, 4])
        b = paddle.randn([4, 5])
        result = paddle.linalg.multi_dot([a, b])
        self.assertEqual(result.shape, [3, 5])


class TestNorm(unittest.TestCase):
    """测试 norm 函数
    Test norm function"""

    def test_norm_frobenius(self):
        """测试 Frobenius 范数
        Test Frobenius norm"""
        a = paddle.randn([3, 4])
        result = paddle.linalg.norm(a)
        self.assertEqual(result.shape, [])

    def test_matrix_norm(self):
        """测试矩阵范数
        Test matrix norm"""
        a = paddle.randn([3, 4])
        result = paddle.linalg.matrix_norm(a)
        self.assertEqual(result.shape, [])

    def test_vector_norm(self):
        """测试向量范数
        Test vector norm"""
        a = paddle.randn([3, 4])
        result = paddle.linalg.vector_norm(a)
        self.assertEqual(result.shape, [])

    def test_vector_norm_l1(self):
        """测试 L1 向量范数
        Test L1 vector norm"""
        a = paddle.to_tensor([1.0, -2.0, 3.0])
        result = paddle.linalg.vector_norm(a, p=1)
        self.assertAlmostEqual(result.item(), 6.0, places=5)

    def test_vector_norm_inf(self):
        """测试无穷向量范数
        Test infinity vector norm"""
        a = paddle.to_tensor([1.0, -3.0, 2.0])
        result = paddle.linalg.vector_norm(a, p=float('inf'))
        self.assertAlmostEqual(result.item(), 3.0, places=5)


class TestMatrixTranspose(unittest.TestCase):
    """测试 matrix_transpose 函数
    Test matrix_transpose function"""

    def test_matrix_transpose_basic(self):
        """测试基本矩阵转置
        Test basic matrix transpose"""
        a = paddle.randn([2, 3, 4])
        result = paddle.linalg.matrix_transpose(a)
        self.assertEqual(result.shape, [2, 4, 3])

    def test_matrix_transpose_2d(self):
        """测试二维矩阵转置
        Test 2D matrix transpose"""
        a = paddle.randn([3, 4])
        result = paddle.linalg.matrix_transpose(a)
        self.assertEqual(result.shape, [4, 3])


class TestHouseholderProduct(unittest.TestCase):
    """测试 householder_product 函数
    Test householder_product function"""

    def test_householder_product_basic(self):
        """测试基本 Householder 乘积
        Test basic Householder product"""
        a = paddle.randn([4, 3])
        tau = paddle.ones([3])
        q = paddle.linalg.householder_product(a, tau)
        self.assertEqual(q.shape, [4, 3])


class TestCdist(unittest.TestCase):
    """测试 cdist 函数
    Test cdist function"""

    def test_cdist_basic(self):
        """测试基本成对距离
        Test basic pairwise distance"""
        a = paddle.randn([3, 2])
        b = paddle.randn([4, 2])
        result = paddle.cdist(a, b)
        self.assertEqual(result.shape, [3, 4])

    def test_cdist_self(self):
        """测试自身成对距离
        Test self pairwise distance"""
        a = paddle.randn([3, 2])
        result = paddle.cdist(a, a)
        self.assertEqual(result.shape, [3, 3])
        # Diagonal should be 0
        diag = paddle.diag(result)
        np.testing.assert_allclose(diag.numpy(), np.zeros(3), atol=1e-5)


class TestCorrcoef(unittest.TestCase):
    """测试 corrcoef 函数
    Test corrcoef function"""

    def test_corrcoef_basic(self):
        """测试基本相关系数矩阵
        Test basic correlation coefficient matrix"""
        a = paddle.randn([3, 5])
        result = paddle.linalg.corrcoef(a)
        self.assertEqual(result.shape, [3, 3])

    def test_corrcoef_diag(self):
        """测试相关系数矩阵对角线为1
        Test correlation matrix diagonal is 1"""
        a = paddle.randn([3, 100])
        result = paddle.linalg.corrcoef(a)
        diag = paddle.diag(result)
        np.testing.assert_allclose(diag.numpy(), np.ones(3), atol=1e-4)


class TestCov(unittest.TestCase):
    """测试 cov 函数
    Test cov function"""

    def test_cov_basic(self):
        """测试基本协方差矩阵
        Test basic covariance matrix"""
        a = paddle.randn([3, 5])
        result = paddle.linalg.cov(a)
        self.assertEqual(result.shape, [3, 3])

    def test_cov_symmetric(self):
        """测试协方差矩阵对称性
        Test covariance matrix symmetry"""
        a = paddle.randn([3, 50])
        result = paddle.linalg.cov(a)
        np.testing.assert_allclose(result.numpy(), result.numpy().T, atol=1e-5)


class TestHistogram(unittest.TestCase):
    """测试 histogram 函数
    Test histogram function"""

    def test_histogram_basic(self):
        """测试基本直方图
        Test basic histogram"""
        a = paddle.randn([100])
        result = paddle.histogram(a, bins=10, min=-3.0, max=3.0)
        self.assertEqual(result.shape, [10])

    def test_histogram_bin_edges(self):
        """测试直方图 bin 边界
        Test histogram bin edges"""
        a = paddle.randn([100])
        edges = paddle.histogram_bin_edges(a, bins=10)
        self.assertEqual(edges.shape, [11])


class TestVecdot(unittest.TestCase):
    """测试 vecdot 函数
    Test vecdot function"""

    def test_vecdot_basic(self):
        """测试基本向量点积
        Test basic vector dot product"""
        a = paddle.to_tensor([1.0, 2.0, 3.0])
        b = paddle.to_tensor([4.0, 5.0, 6.0])
        result = paddle.linalg.vecdot(a, b)
        self.assertAlmostEqual(result.item(), 32.0, places=5)


class TestMatrixExp(unittest.TestCase):
    """测试 matrix_exp 函数
    Test matrix_exp function"""

    def test_matrix_exp_zero(self):
        """测试零矩阵的指数为单位矩阵
        Test matrix exponential of zero matrix is identity"""
        a = paddle.zeros([3, 3])
        result = paddle.linalg.matrix_exp(a)
        np.testing.assert_allclose(result.numpy(), np.eye(3), atol=1e-5)

    def test_matrix_exp_shape(self):
        """测试 matrix_exp 输出形状
        Test matrix_exp output shape"""
        a = paddle.randn([3, 3])
        result = paddle.linalg.matrix_exp(a)
        self.assertEqual(result.shape, [3, 3])


class TestLstsq(unittest.TestCase):
    """测试 lstsq 函数
    Test lstsq function"""

    def test_lstsq_basic(self):
        """测试基本最小二乘求解
        Test basic least squares solve"""
        a = paddle.randn([5, 3])
        b = paddle.randn([5, 1])
        result, residuals, rank, singular_values = paddle.linalg.lstsq(a, b)
        self.assertEqual(result.shape, [3, 1])


class TestSvdvals(unittest.TestCase):
    """测试 svdvals 函数
    Test svdvals function"""

    def test_svdvals_basic(self):
        """测试基本奇异值计算
        Test basic singular value computation"""
        a = paddle.randn([4, 3])
        result = paddle.linalg.svdvals(a)
        self.assertEqual(result.shape, [3])

    def test_svdvals_descending(self):
        """测试奇异值降序排列
        Test singular values in descending order"""
        a = paddle.randn([4, 3])
        result = paddle.linalg.svdvals(a)
        # Values should be non-negative and non-increasing
        diffs = result[:-1] - result[1:]
        self.assertTrue(paddle.all(diffs >= -1e-5).item())


class TestSvdLowrank(unittest.TestCase):
    """测试 svd_lowrank 函数
    Test svd_lowrank function"""

    def test_svd_lowrank_basic(self):
        """测试基本低秩 SVD
        Test basic low-rank SVD"""
        a = paddle.randn([5, 4])
        U, S, V = paddle.linalg.svd_lowrank(a, q=3)
        self.assertEqual(U.shape, [5, 3])
        self.assertEqual(S.shape, [3])
        self.assertEqual(V.shape, [4, 3])


class TestPcaLowrank(unittest.TestCase):
    """测试 pca_lowrank 函数
    Test pca_lowrank function"""

    def test_pca_lowrank_basic(self):
        """测试基本低秩 PCA
        Test basic low-rank PCA"""
        a = paddle.randn([5, 4])
        U, S, V = paddle.linalg.pca_lowrank(a, q=3)
        self.assertEqual(U.shape, [5, 3])
        self.assertEqual(S.shape, [3])
        self.assertEqual(V.shape, [4, 3])


class TestTriangularSolve(unittest.TestCase):
    """测试 triangular_solve 函数
    Test triangular_solve function"""

    def test_triangular_solve_lower(self):
        """测试下三角方程组求解
        Test lower triangular solve"""
        a = paddle.randn([3, 3])
        a = paddle.tril(a) + paddle.eye(3) * 5  # Make lower triangular
        b = paddle.randn([3, 1])
        x = paddle.linalg.triangular_solve(a, b)
        self.assertEqual(x.shape, [3, 1])

    def test_triangular_solve_upper(self):
        """测试上三角方程组求解
        Test upper triangular solve"""
        a = paddle.randn([3, 3])
        a = paddle.triu(a) + paddle.eye(3) * 5  # Make upper triangular
        b = paddle.randn([3, 1])
        x = paddle.linalg.triangular_solve(a, b, upper=True)
        self.assertEqual(x.shape, [3, 1])


class TestEigvals(unittest.TestCase):
    """测试 eigvals 函数
    Test eigvals function"""

    def test_eigvals_basic(self):
        """测试基本特征值计算
        Test basic eigenvalue computation"""
        a = paddle.randn([3, 3])
        result = paddle.linalg.eigvals(a)
        self.assertEqual(result.shape, [3])


if __name__ == '__main__':
    unittest.main()
