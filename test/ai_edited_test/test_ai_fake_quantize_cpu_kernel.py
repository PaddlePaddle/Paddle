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
# Target file: paddle/phi/kernels/funcs/fake_quantize_functor.cc
# Tests for fake quantize CPU kernels.
# Exercises the C++ fake quantize functors via paddle._C_ops.fake_quantize_dequantize_abs_max.
#
# 本文件针对 fake_quantize_functor.cc 中的伪量化 CPU 算子编写单元测试。
# 通过 paddle._C_ops.fake_quantize_dequantize_abs_max 调用 C++ 内核，
# 验证伪量化-反量化操作的数值正确性。

import unittest

import numpy as np

import paddle


class TestFakeQuantizeDequantizeAbsMaxCPU(unittest.TestCase):
    """Test fake_quantize_dequantize_abs_max on CPU.
    测试 CPU 上的伪量化-反量化绝对值最大值操作。"""

    def setUp(self):
        paddle.set_device("cpu")

    def tearDown(self):
        pass

    def test_fake_quantize_dequantize_basic(self):
        """Basic fake quantize-dequantize: output should be close to input but quantized.
        基础伪量化-反量化测试：输出应接近输入但经过量化。"""
        x = paddle.to_tensor([1.0, 2.0, 3.0, -4.0, 5.0])
        out, scale = paddle._C_ops.fake_quantize_dequantize_abs_max(x, 8, 1)
        # Scale should be the max absolute value
        self.assertAlmostEqual(scale.numpy()[0], 5.0, places=5)
        # Output should be close to input (quantization error small)
        np.testing.assert_allclose(out.numpy(), x.numpy(), atol=0.1)

    def test_fake_quantize_dequantize_scale_is_absmax(self):
        """Scale should be the max absolute value of the input.
        scale 应为输入的最大绝对值。"""
        x = paddle.to_tensor([1.0, -3.0, 2.0, 5.0, -1.0])
        out, scale = paddle._C_ops.fake_quantize_dequantize_abs_max(x, 8, 1)
        self.assertAlmostEqual(scale.numpy()[0], 5.0, places=5)

    def test_fake_quantize_dequantize_shape_preserved(self):
        """Output shape should match input shape.
        输出形状应与输入形状一致。"""
        x = paddle.randn([2, 3, 4])
        out, scale = paddle._C_ops.fake_quantize_dequantize_abs_max(x, 8, 1)
        self.assertEqual(out.shape, [2, 3, 4])

    def test_fake_quantize_dequantize_low_bit(self):
        """Fake quantize-dequantize with low bit_width=4.
        低 bit_width=4 的伪量化-反量化测试。"""
        x = paddle.to_tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        out, scale = paddle._C_ops.fake_quantize_dequantize_abs_max(x, 4, 1)
        # With 4 bits, quantization error is larger
        self.assertAlmostEqual(scale.numpy()[0], 5.0, places=5)
        # Output should still be reasonable
        np.testing.assert_allclose(out.numpy(), x.numpy(), atol=0.5)

    def test_fake_quantize_dequantize_high_bit(self):
        """Fake quantize-dequantize with high bit_width=16: nearly lossless.
        高 bit_width=16 的伪量化-反量化测试：几乎无损。"""
        x = paddle.to_tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        out, scale = paddle._C_ops.fake_quantize_dequantize_abs_max(x, 16, 1)
        np.testing.assert_allclose(out.numpy(), x.numpy(), atol=0.01)

    def test_fake_quantize_dequantize_all_zeros(self):
        """Fake quantize-dequantize with all-zero input.
        全零输入的伪量化-反量化测试。"""
        x = paddle.zeros([5])
        out, scale = paddle._C_ops.fake_quantize_dequantize_abs_max(x, 8, 1)
        np.testing.assert_array_almost_equal(
            out.numpy(), [0.0, 0.0, 0.0, 0.0, 0.0]
        )

    def test_fake_quantize_dequantize_negative_values(self):
        """Fake quantize-dequantize with all negative values.
        全负值输入的伪量化-反量化测试。"""
        x = paddle.to_tensor([-1.0, -2.0, -3.0])
        out, scale = paddle._C_ops.fake_quantize_dequantize_abs_max(x, 8, 1)
        self.assertAlmostEqual(scale.numpy()[0], 3.0, places=5)
        np.testing.assert_allclose(out.numpy(), x.numpy(), atol=0.1)

    def test_fake_quantize_dequantize_large_tensor(self):
        """Fake quantize-dequantize with large tensor.
        大规模张量的伪量化-反量化测试。"""
        x = paddle.randn([100, 100])
        out, scale = paddle._C_ops.fake_quantize_dequantize_abs_max(x, 8, 1)
        self.assertEqual(out.shape, [100, 100])
        # Scale should be the max absolute value
        self.assertAlmostEqual(
            scale.numpy()[0], paddle.abs(x).max().item(), places=4
        )
        # Output should be reasonably close to input
        max_error = paddle.abs(out - x).max().item()
        self.assertLess(max_error, 0.1)

    def test_fake_quantize_dequantize_single_value(self):
        """Fake quantize-dequantize with single value: should be exact.
        单值输入的伪量化-反量化测试：应该精确。"""
        x = paddle.to_tensor([3.0])
        out, scale = paddle._C_ops.fake_quantize_dequantize_abs_max(x, 8, 1)
        self.assertAlmostEqual(out.numpy()[0], 3.0, places=4)


class TestFakeDequantizeMaxAbsCPU(unittest.TestCase):
    """Test fake_dequantize_max_abs on CPU.
    测试 CPU 上的伪反量化绝对值最大值操作。"""

    def setUp(self):
        paddle.set_device("cpu")

    def tearDown(self):
        pass

    def test_fake_dequantize_basic(self):
        """Basic dequantize: out = in * scale / max_range.
        基础反量化测试：out = in * scale / max_range。"""
        x = paddle.to_tensor([1.0, 2.0, 3.0])
        scale = paddle.to_tensor([5.0])
        max_range = 127.0
        out = paddle._C_ops.fake_dequantize_max_abs(x, scale, max_range)
        expected = x.numpy() * 5.0 / 127.0
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5)

    def test_fake_dequantize_shape(self):
        """Dequantize output shape should match input shape.
        反量化输出形状应与输入形状一致。"""
        x = paddle.randn([3, 4, 5])
        scale = paddle.to_tensor([1.0])
        out = paddle._C_ops.fake_dequantize_max_abs(x, scale, 255.0)
        self.assertEqual(out.shape, [3, 4, 5])

    def test_fake_dequantize_different_max_range(self):
        """Dequantize with different max_range values.
        使用不同 max_range 值的反量化测试。"""
        x = paddle.to_tensor([100.0])
        scale = paddle.to_tensor([1.0])
        for max_range in [127.0, 255.0, 65535.0]:
            out = paddle._C_ops.fake_dequantize_max_abs(x, scale, max_range)
            expected = 100.0 / max_range
            np.testing.assert_almost_equal(out.numpy()[0], expected)


if __name__ == "__main__":
    unittest.main()
