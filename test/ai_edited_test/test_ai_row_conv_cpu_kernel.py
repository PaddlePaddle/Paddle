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
# Target file: paddle/phi/kernels/cpu/row_conv_kernel.cc
# Tests for row_conv CPU kernel.
# Exercises the C++ RowConvKernel via paddle.nn.functional.row_conv or paddle._C_ops.
# Note: row_conv may only be available in static graph mode or via paddle.nn.RowConv.
#
# 本文件针对 row_conv_kernel.cc 中的行卷积 CPU 算子编写单元测试。
# 通过 paddle API 或 _C_ops 调用 C++ RowConvKernel。
# 注意：row_conv 可能在静态图模式下或通过 paddle.nn.RowConv 使用。

import unittest

import paddle


class TestRowConvCPU(unittest.TestCase):
    """Test row_conv on CPU.
    测试 CPU 上的行卷积操作。"""

    def setUp(self):
        paddle.set_device("cpu")

    def tearDown(self):
        pass

    def _get_row_conv_fn(self):
        """Get row_conv function from available APIs.
        从可用的 API 获取 row_conv 函数。"""
        # Try paddle.nn.RowConv layer
        if hasattr(paddle.nn, "RowConv"):
            return "layer", paddle.nn.RowConv
        # Try _C_ops
        if hasattr(paddle._C_ops, "row_conv"):
            return "c_ops", paddle._C_ops.row_conv
        # Try paddle.nn.functional
        if hasattr(paddle.nn.functional, "row_conv"):
            return "functional", paddle.nn.functional.row_conv
        return None, None

    def test_row_conv_api_available(self):
        """Check if row_conv is available via nn.Layer.
        检查 row_conv 是否通过 nn.Layer 可用。"""
        source, _ = self._get_row_conv_fn()
        # row_conv may or may not be in the public API for eager mode
        # We skip this test if not available
        if source is None:
            self.skipTest("row_conv API not available in this build")

    def test_row_conv_basic(self):
        """Basic row_conv test: computes future context weighted sum.
        基础 row_conv 测试：计算未来上下文加权求和。
        out[b, t, d] = sum_{w=0}^{future_context-1} filter[w, d] * x[b, t+w, d]"""
        source, fn = self._get_row_conv_fn()
        if source is None:
            self.skipTest("row_conv API not available")

        if source == "layer":
            # Use paddle.nn.RowConv
            x = paddle.to_tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
            layer = paddle.nn.RowConv(2, 2)  # (input_dim, future_context)
            result = layer(x)
            self.assertEqual(result.shape, [1, 3, 2])
        elif source == "c_ops":
            x = paddle.to_tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
            w = paddle.to_tensor([[0.5, 1.0], [0.3, 0.7]])
            result = fn(x, w)
            self.assertEqual(result.shape, [1, 3, 2])
        elif source == "functional":
            x = paddle.to_tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
            w = paddle.to_tensor([[0.5, 1.0], [0.3, 0.7]])
            result = fn(x, w)
            self.assertEqual(result.shape, [1, 3, 2])

    def test_row_conv_output_shape(self):
        """Row conv output shape should match input shape [batch, timesteps, dim].
        行卷积输出形状应与输入形状 [batch, timesteps, dim] 一致。"""
        source, fn = self._get_row_conv_fn()
        if source is None:
            self.skipTest("row_conv API not available")

        if source == "layer":
            x = paddle.randn([4, 10, 8])
            layer = paddle.nn.RowConv(8, 3)
            result = layer(x)
            self.assertEqual(result.shape, [4, 10, 8])
        elif source in ("c_ops", "functional"):
            x = paddle.randn([4, 10, 8])
            w = paddle.randn([3, 8])
            result = fn(x, w)
            self.assertEqual(result.shape, [4, 10, 8])

    def test_row_conv_single_timestep(self):
        """Row conv with single timestep.
        单时间步的行卷积测试。"""
        source, fn = self._get_row_conv_fn()
        if source is None:
            self.skipTest("row_conv API not available")

        if source == "layer":
            x = paddle.randn([2, 1, 4])
            layer = paddle.nn.RowConv(4, 2)
            result = layer(x)
            self.assertEqual(result.shape, [2, 1, 4])
        elif source in ("c_ops", "functional"):
            x = paddle.randn([2, 1, 4])
            w = paddle.randn([2, 4])
            result = fn(x, w)
            self.assertEqual(result.shape, [2, 1, 4])

    def test_row_conv_no_nan(self):
        """Row conv should not produce NaN values.
        行卷积不应产生 NaN 值。"""
        source, fn = self._get_row_conv_fn()
        if source is None:
            self.skipTest("row_conv API not available")

        if source == "layer":
            x = paddle.randn([3, 8, 16])
            layer = paddle.nn.RowConv(16, 4)
            result = layer(x)
        elif source in ("c_ops", "functional"):
            x = paddle.randn([3, 8, 16])
            w = paddle.randn([4, 16])
            result = fn(x, w)

        self.assertFalse(paddle.any(paddle.isnan(result)))


if __name__ == "__main__":
    unittest.main()
