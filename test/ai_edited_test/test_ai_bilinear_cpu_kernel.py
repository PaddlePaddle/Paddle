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
# Target file: paddle/phi/kernels/cpu/bilinear_kernel.cc
# Tests for bilinear CPU kernel.
# Exercises the C++ BilinearKernel via paddle.nn.functional.bilinear API.
# Note: Paddle's bilinear weight shape is [out_features, in1_features, in2_features].
#
# 本文件针对 bilinear_kernel.cc 中的双线性变换 CPU 算子编写单元测试。
# 通过 paddle.nn.functional.bilinear API 来调用 C++ 内核，验证双线性变换的正确性。
# 注意：Paddle 的 bilinear 权重形状为 [out_features, in1_features, in2_features]。

import unittest

import numpy as np

import paddle
import paddle.nn.functional as F


class TestBilinearCPU(unittest.TestCase):
    """Test bilinear transformation on CPU.
    测试 CPU 上的双线性变换操作。"""

    def setUp(self):
        paddle.set_device("cpu")

    def tearDown(self):
        pass

    def test_bilinear_basic(self):
        """Basic bilinear: out = x1 * W^T * x2 + bias.
        基础双线性变换测试：out = x1 * W^T * x2 + bias。
        Weight shape: [out_features, in1_features, in2_features]."""
        x1 = paddle.randn([2, 3])
        x2 = paddle.randn([2, 4])
        w = paddle.randn([5, 3, 4])  # [out, in1, in2]
        b = paddle.randn([1, 5])  # bias must be [1, out_features]
        result = F.bilinear(x1, x2, w, b)
        self.assertEqual(result.shape, [2, 5])
        # Verify correctness against manual computation
        x1_np = x1.numpy()
        x2_np = x2.numpy()
        w_np = w.numpy()
        b_np = b.numpy()
        expected = np.zeros((2, 5), dtype="float32")
        for i in range(2):
            for k in range(5):
                val = 0.0
                for j in range(3):
                    for l in range(4):
                        val += x1_np[i, j] * w_np[k, j, l] * x2_np[i, l]
                expected[i, k] = val + b_np[0, k]
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

    def test_bilinear_no_bias(self):
        """Bilinear without bias.
        无偏置的双线性变换测试。"""
        x1 = paddle.randn([3, 4])
        x2 = paddle.randn([3, 5])
        w = paddle.randn([6, 4, 5])
        result = F.bilinear(x1, x2, w)
        self.assertEqual(result.shape, [3, 6])

    def test_bilinear_single_sample(self):
        """Bilinear with single sample.
        单样本的双线性变换测试。"""
        x1 = paddle.randn([1, 3])
        x2 = paddle.randn([1, 4])
        w = paddle.randn([2, 3, 4])
        b = paddle.randn([1, 2])
        result = F.bilinear(x1, x2, w, b)
        self.assertEqual(result.shape, [1, 2])

    def test_bilinear_float64(self):
        """Bilinear with float64 dtype.
        float64 数据类型的双线性变换测试。"""
        x1 = paddle.randn([2, 3], dtype="float64")
        x2 = paddle.randn([2, 4], dtype="float64")
        w = paddle.randn([5, 3, 4], dtype="float64")
        b = paddle.randn([1, 5], dtype="float64")
        result = F.bilinear(x1, x2, w, b)
        self.assertEqual(result.dtype, paddle.float64)
        self.assertEqual(result.shape, [2, 5])

    def test_bilinear_zero_bias(self):
        """Bilinear with zero bias.
        偏置为零的双线性变换测试。"""
        x1 = paddle.randn([2, 3])
        x2 = paddle.randn([2, 4])
        w = paddle.randn([5, 3, 4])
        b = paddle.zeros([1, 5])
        result = F.bilinear(x1, x2, w, b)
        # Verify against no-bias version
        result_no_bias = F.bilinear(x1, x2, w)
        np.testing.assert_allclose(
            result.numpy(), result_no_bias.numpy(), rtol=1e-6
        )

    def test_bilinear_large(self):
        """Bilinear with larger dimensions.
        更大维度的双线性变换测试。"""
        x1 = paddle.randn([10, 8])
        x2 = paddle.randn([10, 8])
        w = paddle.randn([16, 8, 8])
        b = paddle.randn([1, 16])
        result = F.bilinear(x1, x2, w, b)
        self.assertEqual(result.shape, [10, 16])

    def test_bilinear_zero_inputs(self):
        """Bilinear with zero inputs should produce bias.
        输入为零的双线性变换应产生偏置结果。"""
        x1 = paddle.zeros([2, 3])
        x2 = paddle.zeros([2, 4])
        w = paddle.randn([5, 3, 4])
        b = paddle.randn([1, 5])
        result = F.bilinear(x1, x2, w, b)
        expected = np.tile(b.numpy(), (2, 1))
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-6)

    def test_bilinear_output_shape(self):
        """Bilinear output shape matches [batch, out_features].
        双线性变换输出形状为 [batch, out_features]。"""
        for batch in [1, 5, 10]:
            for out_f in [1, 4, 16]:
                x1 = paddle.randn([batch, 3])
                x2 = paddle.randn([batch, 4])
                w = paddle.randn([out_f, 3, 4])
                result = F.bilinear(x1, x2, w)
                self.assertEqual(result.shape, [batch, out_f])

    def test_bilinear_different_in_dims(self):
        """Bilinear with different input dimensions.
        不同输入维度的双线性变换测试。"""
        x1 = paddle.randn([3, 5])
        x2 = paddle.randn([3, 7])
        w = paddle.randn([4, 5, 7])
        result = F.bilinear(x1, x2, w)
        self.assertEqual(result.shape, [3, 4])


if __name__ == "__main__":
    unittest.main()
