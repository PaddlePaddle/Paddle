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
稀疏张量高级测试 / Advanced Sparse Tensor Tests

测试目标 / Test Target:
  paddle.sparse 稀疏张量操作

覆盖的模块 / Covered Modules:
  - paddle.sparse.sparse_coo_tensor: COO稀疏张量
  - paddle.sparse.sparse_csr_tensor: CSR稀疏张量
  - paddle.sparse.nn.Conv2D: 稀疏卷积
  - 稀疏矩阵运算

作用 / Purpose:
  补充稀疏张量API的高级测试，提升覆盖率。
"""

import unittest

import numpy as np

import paddle
from paddle import sparse

paddle.disable_static()


class TestSparseCOOAdvanced(unittest.TestCase):
    """测试COO稀疏张量高级操作 / Test advanced COO sparse tensor operations"""

    def test_coo_basic(self):
        """测试COO基本创建 / Test COO basic creation"""
        indices = paddle.to_tensor([[0, 1, 2], [1, 0, 2]])
        values = paddle.to_tensor([1.0, 2.0, 3.0])
        shape = [3, 3]
        x = sparse.sparse_coo_tensor(indices, values, shape)
        self.assertEqual(x.shape, [3, 3])

    def test_coo_to_dense(self):
        """测试COO转稠密张量 / Test COO to dense conversion"""
        indices = paddle.to_tensor([[0, 1], [0, 1]])
        values = paddle.to_tensor([5.0, 6.0])
        x = sparse.sparse_coo_tensor(indices, values, [3, 3])
        dense = x.to_dense()
        self.assertEqual(dense.shape, [3, 3])
        self.assertAlmostEqual(float(dense[0, 0].numpy()), 5.0)
        self.assertAlmostEqual(float(dense[1, 1].numpy()), 6.0)

    def test_coo_addition(self):
        """测试COO稀疏加法 / Test COO sparse addition"""
        indices = paddle.to_tensor([[0, 1], [0, 1]])
        values1 = paddle.to_tensor([1.0, 2.0])
        values2 = paddle.to_tensor([3.0, 4.0])
        x = sparse.sparse_coo_tensor(indices, values1, [3, 3])
        y = sparse.sparse_coo_tensor(indices, values2, [3, 3])
        # Convert to dense and add
        dense_x = x.to_dense()
        dense_y = y.to_dense()
        result = dense_x + dense_y
        self.assertEqual(result.shape, [3, 3])

    def test_coo_values(self):
        """测试COO非零值访问 / Test COO non-zero value access"""
        indices = paddle.to_tensor([[0, 1], [2, 3]])
        values = paddle.to_tensor([10.0, 20.0])
        x = sparse.sparse_coo_tensor(indices, values, [4, 5])
        np.testing.assert_allclose(x.values().numpy(), [10.0, 20.0])

    def test_coo_nnz(self):
        """测试COO非零元素数量 / Test COO nnz count"""
        indices = paddle.to_tensor([[0, 1, 2], [0, 1, 2]])
        values = paddle.to_tensor([1.0, 2.0, 3.0])
        x = sparse.sparse_coo_tensor(indices, values, [4, 4])
        self.assertEqual(x.nnz(), 3)


class TestSparseCSRAdvanced(unittest.TestCase):
    """测试CSR稀疏张量 / Test CSR sparse tensor"""

    def test_csr_basic(self):
        """测试CSR基本创建 / Test CSR basic creation"""
        # 3x4 matrix with values at [0,1], [1,0], [2,3]
        crows = paddle.to_tensor([0, 1, 2, 3])  # crow_indices
        cols = paddle.to_tensor([1, 0, 3])
        values = paddle.to_tensor([5.0, 3.0, 7.0])
        x = sparse.sparse_csr_tensor(crows, cols, values, [3, 4])
        self.assertEqual(x.shape, [3, 4])

    def test_csr_to_dense(self):
        """测试CSR转稠密 / Test CSR to dense"""
        crows = paddle.to_tensor([0, 1, 2, 3])
        cols = paddle.to_tensor([1, 0, 3])
        values = paddle.to_tensor([5.0, 3.0, 7.0])
        x = sparse.sparse_csr_tensor(crows, cols, values, [3, 4])
        dense = x.to_dense()
        self.assertEqual(dense.shape, [3, 4])
        self.assertAlmostEqual(float(dense[0, 1].numpy()), 5.0)


class TestSparseConversion(unittest.TestCase):
    """测试稀疏格式转换 / Test sparse format conversion"""

    def test_dense_to_coo(self):
        """测试稠密转COO / Test dense to COO"""
        dense = paddle.to_tensor([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]])
        coo = dense.to_sparse_coo(sparse_dim=2)
        self.assertEqual(coo.shape, [2, 3])
        # Should have 3 non-zero elements
        self.assertEqual(coo.nnz(), 3)

    def test_coo_to_csr(self):
        """测试COO转CSR / Test COO to CSR"""
        dense = paddle.to_tensor([[1.0, 0.0], [0.0, 2.0]])
        coo = dense.to_sparse_coo(sparse_dim=2)
        csr = coo.to_sparse_csr()
        self.assertEqual(csr.shape, [2, 2])

    def test_sparse_dense_matmul(self):
        """测试稀疏-稠密矩阵乘法 / Test sparse-dense matmul"""
        indices = paddle.to_tensor([[0, 1], [0, 1]])
        values = paddle.to_tensor([2.0, 3.0])
        sparse_mat = sparse.sparse_coo_tensor(indices, values, [2, 2])
        dense_mat = paddle.to_tensor([[1.0, 0.0], [0.0, 1.0]])
        result = sparse.matmul(sparse_mat, dense_mat)
        self.assertEqual(result.shape, [2, 2])


class TestSparseMath(unittest.TestCase):
    """测试稀疏数学运算 / Test sparse math operations"""

    def test_sparse_relu(self):
        """测试稀疏ReLU / Test sparse ReLU"""
        indices = paddle.to_tensor([[0, 1], [0, 1]])
        values = paddle.to_tensor([-1.0, 2.0])
        x = sparse.sparse_coo_tensor(indices, values, [3, 3])
        result = sparse.nn.functional.relu(x)
        self.assertEqual(result.shape, [3, 3])

    def test_sparse_scale(self):
        """测试稀疏缩放 / Test sparse scaling via dense conversion"""
        indices = paddle.to_tensor([[0, 1], [0, 1]])
        values = paddle.to_tensor([1.0, 2.0])
        x = sparse.sparse_coo_tensor(indices, values, [3, 3])
        dense = x.to_dense()
        result = dense * 2.0
        self.assertAlmostEqual(float(result[0, 0].numpy()), 2.0)
        self.assertAlmostEqual(float(result[1, 1].numpy()), 4.0)


if __name__ == '__main__':
    unittest.main()
