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

# [AUTO-GENERATED] Do not edit manually.
# Target source: paddle/phi/kernels/cpu/dropout_kernel.cc
# Generated for exercising C++ CPU kernel: DropoutRawKernel, DropoutNdKernel
#
# 测试 Dropout CPU 内核
# Tests for Dropout CPU kernel

import unittest

import numpy as np

import paddle
import paddle.nn.functional as F


class TestDropoutRawKernelBasic(unittest.TestCase):
    """基本 Dropout 测试 / Basic Dropout tests"""

    def setUp(self):
        """初始化测试环境 / Setup test environment"""
        paddle.disable_static()
        paddle.set_device("cpu")

    def tearDown(self):
        """清理测试环境 / Teardown test environment"""
        paddle.enable_static()

    def test_dropout_zero_prob(self):
        """测试 dropout_prob=0 时输出等于输入
        Test that dropout with p=0 returns input unchanged
        """
        x_np = np.random.randn(4, 8).astype("float32")
        x = paddle.to_tensor(x_np)
        out = F.dropout(x, p=0.0, training=True)
        np.testing.assert_allclose(out.numpy(), x_np, atol=1e-6)

    def test_dropout_one_prob(self):
        """测试 dropout_prob=1 时输出全为零
        Test that dropout with p=1 returns all zeros
        """
        x_np = np.random.randn(4, 8).astype("float32")
        x = paddle.to_tensor(x_np)
        out = F.dropout(x, p=1.0, training=True)
        np.testing.assert_array_equal(out.numpy(), np.zeros_like(x_np))

    def test_dropout_inference_mode(self):
        """测试推理模式下输出不变
        Test that inference mode returns input unchanged
        """
        x_np = np.random.randn(3, 5).astype("float32")
        x = paddle.to_tensor(x_np)
        out = F.dropout(x, p=0.5, training=False)
        np.testing.assert_allclose(out.numpy(), x_np, atol=1e-6)

    def test_dropout_upscale_in_train(self):
        """测试 upscale_in_train 模式下非零元素被缩放
        Test upscale_in_train mode scales non-dropped elements
        """
        np.random.seed(42)
        x_np = np.ones((1000,), dtype="float32")
        x = paddle.to_tensor(x_np)
        out = F.dropout(x, p=0.5, training=True, mode="upscale_in_train")

        out_np = out.numpy()
        nonzero_vals = out_np[out_np != 0]
        zero_vals = out_np[out_np == 0]

        # Non-dropped elements should be scaled by 1/(1-p) = 2.0
        if len(nonzero_vals) > 0:
            np.testing.assert_allclose(
                nonzero_vals, np.full(len(nonzero_vals), 2.0), atol=1e-4
            )
        # Some elements should be zero (dropped)
        self.assertGreater(len(zero_vals), 0)

    def test_dropout_downscale_in_infer(self):
        """测试 downscale_in_infer 训练模式下非零元素为原始值
        Test downscale_in_infer mode in training: non-dropped elements keep original value
        """
        np.random.seed(42)
        x_np = np.ones((1000,), dtype="float32")
        x = paddle.to_tensor(x_np)
        out = F.dropout(x, p=0.5, training=True, mode="downscale_in_infer")

        out_np = out.numpy()
        nonzero_vals = out_np[out_np != 0]

        # Non-dropped elements should keep original value (1.0)
        if len(nonzero_vals) > 0:
            np.testing.assert_allclose(
                nonzero_vals, np.ones(len(nonzero_vals)), atol=1e-5
            )

    def test_dropout_downscale_in_infer_eval(self):
        """测试 downscale_in_infer 推理模式下输出缩放
        Test downscale_in_infer mode in inference: output scaled by (1-p)
        """
        x_np = np.ones((2, 5), dtype="float32")
        x = paddle.to_tensor(x_np)
        out = F.dropout(x, p=0.3, training=False, mode="downscale_in_infer")
        np.testing.assert_allclose(out.numpy(), np.full((2, 5), 0.7), atol=1e-5)

    def test_dropout_fixed_seed_reproducibility(self):
        """测试固定种子产生可重复结果
        Test that fixed seed produces reproducible results
        """
        x = paddle.ones([100], dtype="float32")

        paddle.seed(123)
        out1 = F.dropout(x, p=0.3, training=True)

        paddle.seed(123)
        out2 = F.dropout(x, p=0.3, training=True)

        np.testing.assert_allclose(out1.numpy(), out2.numpy(), atol=1e-6)

    def test_dropout_output_shape(self):
        """测试 Dropout 保持输出形状
        Test that Dropout preserves output shape
        """
        for shape in [(10,), (10, 20), (2, 3, 4), (1, 1, 5, 5)]:
            x = paddle.randn(shape, dtype="float32")
            out = F.dropout(x, p=0.5, training=True)
            self.assertEqual(out.shape, shape)


class TestDropoutNdKernel(unittest.TestCase):
    """DropoutNd 内核测试（通过 axis 参数测试） / DropoutNd kernel tests via axis parameter"""

    def setUp(self):
        paddle.disable_static()
        paddle.set_device("cpu")

    def tearDown(self):
        paddle.enable_static()

    def test_dropout_with_axis(self):
        """测试带 axis 参数的 Dropout（即 DropoutNd 内核）
        Test dropout with axis parameter (i.e. DropoutNd kernel)
        """
        x = paddle.ones([2, 3, 4, 5], dtype="float32")
        out = F.dropout(x, p=0.0, training=True, axis=[0])
        # With p=0, output should equal input
        np.testing.assert_allclose(out.numpy(), x.numpy(), atol=1e-6)

    def test_dropout_with_axis_p1(self):
        """测试带 axis 参数的 Dropout p=1 时输出全零
        Test dropout with axis and p=1 returns all zeros
        """
        x = paddle.randn([3, 6, 6], dtype="float32")
        out = F.dropout(x, p=1.0, training=True, axis=[1])
        np.testing.assert_array_equal(out.numpy(), np.zeros_like(x.numpy()))

    def test_dropout_with_axis_inference(self):
        """测试带 axis 参数的 Dropout 推理模式
        Test dropout with axis in inference mode
        """
        x = paddle.randn([2, 4, 4], dtype="float32")
        out = F.dropout(x, p=0.5, training=False, axis=[0])
        np.testing.assert_allclose(out.numpy(), x.numpy(), atol=1e-6)

    def test_dropout_with_axis_shapes(self):
        """测试带 axis 参数的 Dropout 不同形状
        Test dropout with axis parameter and different shapes
        """
        x = paddle.ones([4, 8, 16], dtype="float32")
        out = F.dropout(x, p=0.5, training=True, axis=[1])
        self.assertEqual(out.shape, (4, 8, 16))


class TestDropoutDtype(unittest.TestCase):
    """Dropout 不同数据类型测试 / Dropout tests with different dtypes"""

    def setUp(self):
        paddle.disable_static()
        paddle.set_device("cpu")

    def tearDown(self):
        paddle.enable_static()

    def test_dropout_float64(self):
        """测试 float64 类型的 Dropout
        Test Dropout with float64
        """
        x = paddle.randn([10, 10], dtype="float64")
        out = F.dropout(x, p=0.5, training=True)
        self.assertEqual(out.dtype, paddle.float64)
        self.assertEqual(out.shape, (10, 10))

    def test_dropout_float16(self):
        """测试 float16 类型的 Dropout
        Test Dropout with float16
        """
        x = paddle.randn([10, 10], dtype="float16")
        out = F.dropout(x, p=0.5, training=True)
        self.assertEqual(out.dtype, paddle.float16)
        self.assertEqual(out.shape, (10, 10))


class TestDropoutEdgeCases(unittest.TestCase):
    """Dropout 边界情况测试 / Dropout edge case tests"""

    def setUp(self):
        paddle.disable_static()
        paddle.set_device("cpu")

    def tearDown(self):
        paddle.enable_static()

    def test_dropout_single_element(self):
        """测试单元素张量的 Dropout
        Test Dropout with single element tensor
        """
        x = paddle.to_tensor([3.14], dtype="float32")
        out = F.dropout(x, p=0.0, training=True)
        np.testing.assert_allclose(out.numpy(), [3.14], atol=1e-6)

    def test_dropout_very_small_prob(self):
        """测试极小 dropout 概率（大部分元素保留）
        Test Dropout with very small probability (most elements preserved)
        """
        np.random.seed(42)
        x_np = np.ones((1000,), dtype="float32")
        x = paddle.to_tensor(x_np)
        out = F.dropout(x, p=1e-6, training=True)
        out_np = out.numpy()
        # With tiny prob, almost everything should be non-zero
        nonzero_count = np.count_nonzero(out_np)
        self.assertGreater(nonzero_count, 950)

    def test_dropout_negative_values(self):
        """测试包含负值张量的 Dropout
        Test Dropout with negative values
        """
        x = paddle.to_tensor([-1.0, 0.0, 1.0, -2.0], dtype="float32")
        out = F.dropout(x, p=0.0, training=True)
        np.testing.assert_allclose(
            out.numpy(), [-1.0, 0.0, 1.0, -2.0], atol=1e-6
        )

    def test_dropout_large_tensor(self):
        """测试大张量的 Dropout
        Test Dropout with large tensor
        """
        x = paddle.randn([100, 200], dtype="float32")
        out = F.dropout(x, p=0.5, training=True)
        self.assertEqual(out.shape, (100, 200))
        # Approximately half should be non-zero
        nonzero_ratio = np.count_nonzero(out.numpy()) / out.numpy().size
        self.assertGreater(nonzero_ratio, 0.3)
        self.assertLess(nonzero_ratio, 0.7)


if __name__ == "__main__":
    unittest.main()
