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
# Target file: paddle/phi/kernels/cpu/bce_loss_kernel.cc
# Tests for BCELoss CPU kernel.
# Exercises the C++ BCELossKernel via paddle.nn.functional.binary_cross_entropy API.
#
# 本文件针对 bce_loss_kernel.cc 中的二元交叉熵损失 CPU 算子编写单元测试。
# 通过 paddle.nn.functional.binary_cross_entropy API 来调用 C++ 内核，
# 验证 BCE 损失计算的数值正确性。

import unittest

import numpy as np

import paddle
import paddle.nn.functional as F


class TestBCELossCPU(unittest.TestCase):
    """Test binary cross entropy loss on CPU.
    测试 CPU 上的二元交叉熵损失计算。"""

    def setUp(self):
        paddle.set_device("cpu")

    def tearDown(self):
        pass

    def test_bce_loss_perfect_prediction(self):
        """BCE loss for perfect prediction: pred=1, label=1 -> loss=0.
        完美预测的 BCE 损失：pred=1, label=1 -> loss=0。"""
        pred = paddle.to_tensor([1.0])
        label = paddle.to_tensor([1.0])
        loss = F.binary_cross_entropy(pred, label, reduction="none")
        np.testing.assert_array_almost_equal(loss.numpy(), [0.0])

    def test_bce_loss_perfect_negative(self):
        """BCE loss for perfect negative prediction: pred=0, label=0 -> loss=0.
        完美负预测的 BCE 损失：pred=0, label=0 -> loss=0。"""
        pred = paddle.to_tensor([0.0])
        label = paddle.to_tensor([0.0])
        loss = F.binary_cross_entropy(pred, label, reduction="none")
        np.testing.assert_array_almost_equal(loss.numpy(), [0.0])

    def test_bce_loss_worst_case(self):
        """BCE loss for worst prediction: pred=0, label=1 -> loss = -ln(0) clamped.
        最差预测的 BCE 损失：pred=0, label=1 -> loss = -ln(0) 被截断。"""
        pred = paddle.to_tensor([1e-10])
        label = paddle.to_tensor([1.0])
        loss = F.binary_cross_entropy(pred, label, reduction="none")
        # Should be large positive value (clamped at -ln(1e-10) ~ 23)
        self.assertGreater(loss.numpy()[0], 20.0)

    def test_bce_loss_manual(self):
        """BCE loss manual verification: BCE = -(label*log(pred) + (1-label)*log(1-pred)).
        手动验证 BCE 损失计算：BCE = -(label*log(pred) + (1-label)*log(1-pred))。"""
        pred = paddle.to_tensor([0.8, 0.2])
        label = paddle.to_tensor([1.0, 0.0])
        loss = F.binary_cross_entropy(pred, label, reduction="none")
        # For pred=0.8, label=1: -(1*log(0.8) + 0*log(0.2)) = -log(0.8)
        expected = np.array([-np.log(0.8), -np.log(0.8)], dtype="float32")
        np.testing.assert_array_almost_equal(loss.numpy(), expected, decimal=5)

    def test_bce_loss_mean_reduction(self):
        """BCE loss with mean reduction.
        使用均值归约的 BCE 损失测试。"""
        pred = paddle.to_tensor([0.8, 0.2, 0.5])
        label = paddle.to_tensor([1.0, 0.0, 1.0])
        loss = F.binary_cross_entropy(pred, label, reduction="mean")
        individual = np.array(
            [-np.log(0.8), -np.log(0.8), -np.log(0.5)], dtype="float32"
        )
        expected = individual.mean()
        np.testing.assert_almost_equal(loss.numpy(), expected, decimal=5)

    def test_bce_loss_sum_reduction(self):
        """BCE loss with sum reduction.
        使用求和归约的 BCE 损失测试。"""
        pred = paddle.to_tensor([0.8, 0.2])
        label = paddle.to_tensor([1.0, 0.0])
        loss = F.binary_cross_entropy(pred, label, reduction="sum")
        expected = -np.log(0.8) + (-np.log(0.8))
        np.testing.assert_almost_equal(loss.numpy(), expected, decimal=5)

    def test_bce_loss_with_weight(self):
        """BCE loss with per-sample weights.
        带有样本权重的 BCE 损失测试。"""
        pred = paddle.to_tensor([0.8, 0.2])
        label = paddle.to_tensor([1.0, 0.0])
        weight = paddle.to_tensor([2.0, 0.5])
        loss = F.binary_cross_entropy(
            pred, label, weight=weight, reduction="none"
        )
        expected = np.array(
            [-np.log(0.8) * 2.0, -np.log(0.8) * 0.5], dtype="float32"
        )
        np.testing.assert_array_almost_equal(loss.numpy(), expected, decimal=5)

    def test_bce_loss_float64(self):
        """BCE loss with float64 dtype.
        float64 数据类型的 BCE 损失测试。"""
        pred = paddle.to_tensor([0.8], dtype="float64")
        label = paddle.to_tensor([1.0], dtype="float64")
        loss = F.binary_cross_entropy(pred, label, reduction="none")
        self.assertEqual(loss.dtype, paddle.float64)
        np.testing.assert_almost_equal(loss.numpy()[0], -np.log(0.8), decimal=8)

    def test_bce_loss_invalid_input_raises(self):
        """BCE loss with input > 1 should raise error.
        输入值大于 1 应当引发错误。"""
        pred = paddle.to_tensor([1.5])
        label = paddle.to_tensor([1.0])
        with self.assertRaises(ValueError):
            F.binary_cross_entropy(pred, label, reduction="none")

    def test_bce_loss_2d(self):
        """BCE loss with 2D input.
        二维输入的 BCE 损失测试。"""
        pred = paddle.to_tensor([[0.8, 0.2], [0.5, 0.9]])
        label = paddle.to_tensor([[1.0, 0.0], [1.0, 1.0]])
        loss = F.binary_cross_entropy(pred, label, reduction="none")
        self.assertEqual(loss.shape, [2, 2])
        # Check all values are non-negative
        self.assertTrue(np.all(loss.numpy() >= 0))

    def test_bce_loss_boundary_values(self):
        """BCE loss at boundary values (0 and 1).
        边界值（0 和 1）的 BCE 损失测试。"""
        # pred at exact boundaries
        pred = paddle.to_tensor([0.0, 1.0])
        label = paddle.to_tensor([0.0, 1.0])
        loss = F.binary_cross_entropy(pred, label, reduction="none")
        np.testing.assert_array_almost_equal(loss.numpy(), [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
