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
# Target file: paddle/phi/kernels/cpu/binomial_kernel.cc
# Tests for binomial CPU kernel.
# Exercises the C++ BinomialKernel via paddle.binomial API.
#
# 本文件针对 binomial_kernel.cc 中的二项分布采样 CPU 算子编写单元测试。
# 通过 paddle.binomial API 来调用 C++ 内核，验证二项分布采样结果的正确性。

import unittest

import numpy as np

import paddle


class TestBinomialCPU(unittest.TestCase):
    """Test binomial sampling on CPU.
    测试 CPU 上的二项分布采样。"""

    def setUp(self):
        paddle.set_device("cpu")

    def tearDown(self):
        pass

    def test_binomial_basic(self):
        """Basic binomial sampling with fixed count and prob.
        使用固定试验次数和概率的基础二项分布采样测试。"""
        count = paddle.to_tensor([10.0, 10.0, 10.0])
        prob = paddle.to_tensor([0.5, 0.5, 0.5])
        result = paddle.binomial(count, prob)
        self.assertEqual(result.shape, [3])
        self.assertEqual(result.dtype, paddle.int64)
        # All values should be between 0 and count
        for v in result.numpy():
            self.assertGreaterEqual(v, 0)
            self.assertLessEqual(v, 10)

    def test_binomial_prob_zero(self):
        """Binomial with probability 0 should always return 0.
        概率为 0 的二项分布应始终返回 0。"""
        count = paddle.to_tensor([100.0, 50.0])
        prob = paddle.to_tensor([0.0, 0.0])
        result = paddle.binomial(count, prob)
        np.testing.assert_array_equal(result.numpy(), [0, 0])

    def test_binomial_prob_one(self):
        """Binomial with probability 1 should always return count.
        概率为 1 的二项分布应始终返回试验次数。"""
        count = paddle.to_tensor([5.0, 10.0])
        prob = paddle.to_tensor([1.0, 1.0])
        result = paddle.binomial(count, prob)
        np.testing.assert_array_equal(result.numpy(), [5, 10])

    def test_binomial_count_one(self):
        """Binomial with count=1 is a Bernoulli trial.
        试验次数为 1 的二项分布等价于伯努利试验。"""
        count = paddle.to_tensor([1.0] * 1000)
        prob = paddle.to_tensor([0.7] * 1000)
        result = paddle.binomial(count, prob)
        # Mean should be approximately 0.7
        mean = result.numpy().astype("float64").mean()
        self.assertAlmostEqual(mean, 0.7, delta=0.1)

    def test_binomial_output_dtype(self):
        """Binomial output dtype should be int64.
        二项分布输出数据类型应为 int64。"""
        count = paddle.to_tensor([10.0])
        prob = paddle.to_tensor([0.5])
        result = paddle.binomial(count, prob)
        self.assertEqual(result.dtype, paddle.int64)

    def test_binomial_float64(self):
        """Binomial with float64 inputs.
        float64 输入的二项分布采样测试。"""
        count = paddle.to_tensor([20.0], dtype="float64")
        prob = paddle.to_tensor([0.3], dtype="float64")
        result = paddle.binomial(count, prob)
        self.assertEqual(result.dtype, paddle.int64)
        self.assertGreaterEqual(result.numpy()[0], 0)
        self.assertLessEqual(result.numpy()[0], 20)

    def test_binomial_2d_input(self):
        """Binomial with 2D input tensors.
        二维输入张量的二项分布采样测试。"""
        count = paddle.to_tensor([[10.0, 10.0], [10.0, 10.0]])
        prob = paddle.to_tensor([[0.1, 0.9], [0.5, 0.5]])
        result = paddle.binomial(count, prob)
        self.assertEqual(result.shape, [2, 2])

    def test_binomial_range(self):
        """Binomial output values should be in [0, count].
        二项分布输出值应在 [0, count] 范围内。"""
        count = paddle.to_tensor([15.0, 20.0, 25.0])
        prob = paddle.to_tensor([0.3, 0.7, 0.5])
        result = paddle.binomial(count, prob)
        for i, v in enumerate(result.numpy()):
            self.assertGreaterEqual(v, 0)
            self.assertLessEqual(v, count.numpy()[i])

    def test_binomial_single_value(self):
        """Binomial with scalar-like single element.
        单元素标量式的二项分布采样测试。"""
        count = paddle.to_tensor([7.0])
        prob = paddle.to_tensor([0.0])
        result = paddle.binomial(count, prob)
        self.assertEqual(result.numpy()[0], 0)


if __name__ == "__main__":
    unittest.main()
