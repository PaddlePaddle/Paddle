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
# Target file: paddle/phi/kernels/cpu/baddbmm_kernel.cc
# Tests for baddbmm CPU kernel.
# Exercises the C++ BaddbmmKernel via paddle.baddbmm API.
#
# 本文件针对 baddbmm_kernel.cc 中的 baddbmm CPU 算子编写单元测试。
# 通过 paddle.baddbmm API 来调用 C++ 内核，验证批矩阵乘加操作的正确性。

import unittest

import numpy as np

import paddle


class TestBaddbmmCPU(unittest.TestCase):
    """Test baddbmm on CPU.
    测试 CPU 上的 baddbmm（批矩阵乘加）操作。"""

    def setUp(self):
        paddle.set_device("cpu")

    def tearDown(self):
        pass

    def test_baddbmm_basic(self):
        """Basic baddbmm: out = beta*input + alpha*batch1*batch2.
        基础 baddbmm 测试：out = beta*input + alpha*batch1*batch2。"""
        x = paddle.randn([2, 3, 4])
        b1 = paddle.randn([2, 3, 5])
        b2 = paddle.randn([2, 5, 4])
        result = paddle.baddbmm(x, b1, b2)
        # Verify output shape
        self.assertEqual(result.shape, [2, 3, 4])
        # Verify correctness against manual computation
        expected = x.numpy() + np.matmul(b1.numpy(), b2.numpy())
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

    def test_baddbmm_with_alpha_beta(self):
        """Baddbmm with custom alpha and beta.
        带有自定义 alpha 和 beta 的 baddbmm 测试。"""
        x = paddle.randn([2, 3, 4])
        b1 = paddle.randn([2, 3, 5])
        b2 = paddle.randn([2, 5, 4])
        alpha = 2.0
        beta = 0.5
        result = paddle.baddbmm(x, b1, b2, alpha=alpha, beta=beta)
        expected = beta * x.numpy() + alpha * np.matmul(b1.numpy(), b2.numpy())
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

    def test_baddbmm_beta_zero(self):
        """Baddbmm with beta=0 (no input addition).
        beta=0 的 baddbmm 测试（不加输入项）。"""
        x = paddle.randn([1, 3, 4])
        b1 = paddle.randn([1, 3, 5])
        b2 = paddle.randn([1, 5, 4])
        result = paddle.baddbmm(x, b1, b2, beta=0.0, alpha=1.0)
        expected = np.matmul(b1.numpy(), b2.numpy())
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

    def test_baddbmm_alpha_zero(self):
        """Baddbmm with alpha=0 (no matrix multiply).
        alpha=0 的 baddbmm 测试（不做矩阵乘法）。"""
        x = paddle.randn([2, 3, 4])
        b1 = paddle.randn([2, 3, 5])
        b2 = paddle.randn([2, 5, 4])
        result = paddle.baddbmm(x, b1, b2, alpha=0.0, beta=1.0)
        expected = x.numpy()
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

    def test_baddbmm_single_batch(self):
        """Baddbmm with batch_size=1.
        batch_size=1 的 baddbmm 测试。"""
        x = paddle.randn([1, 2, 3])
        b1 = paddle.randn([1, 2, 4])
        b2 = paddle.randn([1, 4, 3])
        result = paddle.baddbmm(x, b1, b2)
        self.assertEqual(result.shape, [1, 2, 3])

    def test_baddbmm_float64(self):
        """Baddbmm with float64 dtype.
        float64 数据类型的 baddbmm 测试。"""
        x = paddle.randn([2, 3, 4], dtype="float64")
        b1 = paddle.randn([2, 3, 5], dtype="float64")
        b2 = paddle.randn([2, 5, 4], dtype="float64")
        result = paddle.baddbmm(x, b1, b2)
        self.assertEqual(result.dtype, paddle.float64)
        expected = x.numpy() + np.matmul(b1.numpy(), b2.numpy())
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-10)

    def test_baddbmm_large_batch(self):
        """Baddbmm with large batch size.
        大批次大小的 baddbmm 测试。"""
        x = paddle.randn([10, 4, 6])
        b1 = paddle.randn([10, 4, 8])
        b2 = paddle.randn([10, 8, 6])
        result = paddle.baddbmm(x, b1, b2)
        self.assertEqual(result.shape, [10, 4, 6])
        expected = x.numpy() + np.matmul(b1.numpy(), b2.numpy())
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

    def test_baddbmm_inplace(self):
        """Baddbmm with input tensor modification.
        带有输入张量原地修改的 baddbmm 测试。"""
        x = paddle.randn([2, 3, 4])
        b1 = paddle.randn([2, 3, 5])
        b2 = paddle.randn([2, 5, 4])
        x_copy = x.clone()
        result = paddle.baddbmm(x, b1, b2)
        # Input should not be modified
        np.testing.assert_array_equal(x.numpy(), x_copy.numpy())


if __name__ == "__main__":
    unittest.main()
