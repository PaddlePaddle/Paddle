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
# Target file: paddle/phi/kernels/funcs/fake_dequantize_functor.cc
# Tests for fake dequantize CPU kernels.
# Exercises the C++ DequantizeFunctor and ChannelDequantizeFunctor via paddle._C_ops.
#
# 本文件针对 fake_dequantize_functor.cc 中的伪反量化 CPU 算子编写单元测试。
# 通过 paddle._C_ops.fake_dequantize_max_abs 调用 C++ DequantizeFunctor，
# 验证伪反量化操作的数值正确性。

import unittest

import numpy as np

import paddle


class TestFakeDequantizeMaxAbsCPU(unittest.TestCase):
    """Test fake_dequantize_max_abs (DequantizeFunctor) on CPU.
    测试 CPU 上的伪反量化（DequantizeFunctor）操作。"""

    def setUp(self):
        paddle.set_device("cpu")

    def tearDown(self):
        pass

    def test_dequantize_formula(self):
        """Verify formula: out = in * scale / max_range.
        验证公式：out = in * scale / max_range。"""
        x = paddle.to_tensor([10.0, 20.0, 30.0])
        scale = paddle.to_tensor([2.0])
        max_range = 127.0
        out = paddle._C_ops.fake_dequantize_max_abs(x, scale, max_range)
        expected = np.array([10.0, 20.0, 30.0]) * 2.0 / 127.0
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-6)

    def test_dequantize_scale_one(self):
        """Dequantize with scale=1.0: out = in / max_range.
        scale=1.0 的反量化：out = in / max_range。"""
        x = paddle.to_tensor([50.0, 100.0])
        scale = paddle.to_tensor([1.0])
        out = paddle._C_ops.fake_dequantize_max_abs(x, scale, 127.0)
        expected = np.array([50.0, 100.0]) / 127.0
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-6)

    def test_dequantize_max_range_255(self):
        """Dequantize with max_range=255 (8-bit).
        max_range=255（8位）的反量化测试。"""
        x = paddle.to_tensor([100.0, 200.0])
        scale = paddle.to_tensor([3.0])
        out = paddle._C_ops.fake_dequantize_max_abs(x, scale, 255.0)
        expected = np.array([100.0, 200.0]) * 3.0 / 255.0
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-6)

    def test_dequantize_2d_tensor(self):
        """Dequantize with 2D tensor.
        二维张量的反量化测试。"""
        x = paddle.randn([4, 5])
        scale = paddle.to_tensor([1.5])
        max_range = 127.0
        out = paddle._C_ops.fake_dequantize_max_abs(x, scale, max_range)
        expected = x.numpy() * 1.5 / 127.0
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-6)

    def test_dequantize_preserves_shape(self):
        """Dequantize output preserves input shape.
        反量化输出保持输入形状。"""
        x = paddle.randn([2, 3, 4, 5])
        scale = paddle.to_tensor([1.0])
        out = paddle._C_ops.fake_dequantize_max_abs(x, scale, 127.0)
        self.assertEqual(out.shape, [2, 3, 4, 5])

    def test_dequantize_preserves_dtype(self):
        """Dequantize output has same dtype as input.
        反量化输出与输入数据类型一致。"""
        x = paddle.randn([3], dtype="float64")
        scale = paddle.to_tensor([1.0], dtype="float64")
        out = paddle._C_ops.fake_dequantize_max_abs(x, scale, 127.0)
        self.assertEqual(out.dtype, paddle.float64)

    def test_dequantize_zero_scale(self):
        """Dequantize with scale=0 should produce all zeros.
        scale=0 的反量化应产生全零。"""
        x = paddle.to_tensor([10.0, 20.0, 30.0])
        scale = paddle.to_tensor([0.0])
        out = paddle._C_ops.fake_dequantize_max_abs(x, scale, 127.0)
        np.testing.assert_array_almost_equal(out.numpy(), [0.0, 0.0, 0.0])

    def test_dequantize_zero_input(self):
        """Dequantize with zero input should produce zero output.
        零输入的反量化应产生零输出。"""
        x = paddle.zeros([5])
        scale = paddle.to_tensor([5.0])
        out = paddle._C_ops.fake_dequantize_max_abs(x, scale, 127.0)
        np.testing.assert_array_almost_equal(out.numpy(), [0.0] * 5)

    def test_dequantize_large_scale(self):
        """Dequantize with large scale value.
        大 scale 值的反量化测试。"""
        x = paddle.to_tensor([1.0])
        scale = paddle.to_tensor([1000.0])
        out = paddle._C_ops.fake_dequantize_max_abs(x, scale, 127.0)
        expected = 1.0 * 1000.0 / 127.0
        np.testing.assert_almost_equal(out.numpy()[0], expected, decimal=5)


class TestFakeDequantizeQuantizeRoundTripCPU(unittest.TestCase):
    """Test round-trip: quantize then dequantize preserves approximate values.
    测试往返过程：量化后反量化应保留近似值。"""

    def setUp(self):
        paddle.set_device("cpu")

    def tearDown(self):
        pass

    def test_round_trip_small_values(self):
        """Quantize-dequantize round trip for small values.
        小值的量化-反量化往返测试。"""
        original = paddle.to_tensor([0.1, 0.5, -0.3, 0.7, -0.9])
        # Quantize-dequantize
        quantized, scale = paddle._C_ops.fake_quantize_dequantize_abs_max(
            original, 8, 1
        )
        # The output should be close to original for small values
        np.testing.assert_allclose(
            quantized.numpy(), original.numpy(), atol=0.05
        )

    def test_round_trip_large_range(self):
        """Quantize-dequantize round trip for large range values.
        大范围值的量化-反量化往返测试。"""
        original = paddle.to_tensor([-100.0, -50.0, 0.0, 50.0, 100.0])
        quantized, scale = paddle._C_ops.fake_quantize_dequantize_abs_max(
            original, 8, 1
        )
        self.assertAlmostEqual(scale.numpy()[0], 100.0, places=5)
        # Output should preserve sign and approximate magnitude
        signs_match = (
            np.sign(quantized.numpy()) == np.sign(original.numpy())
        ).all()
        self.assertTrue(signs_match)


if __name__ == "__main__":
    unittest.main()
