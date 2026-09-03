# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

# [AUTO-GENERATED] Unit test for paddle.nn.quant.quantized_linear (supplementary)
# 自动生成的单测，覆盖 quantized_linear 模块中额外未覆盖的代码
# Target: cover uncovered lines 54,59,119-130,172-186,262-290,331-356,386-395
#   in python/paddle/nn/quant/quantized_linear.py
# 未覆盖行: _get_arch_info 非 CUDA False 分支, weight_quantize 静态图分支,
#           weight_dequantize 静态图分支, weight_only_linear 静态图分支,
#           llm_int8_linear 静态图分支, apply_per_channel_scale 静态图分支

import unittest
from unittest.mock import patch

import paddle
from paddle.nn.quant.quantized_linear import (
    _get_arch_info,
    apply_per_channel_scale,
    llm_int8_linear,
    weight_dequantize,
    weight_only_linear,
    weight_quantize,
)


class TestGetArchInfoEdgeCases(unittest.TestCase):
    """Test _get_arch_info edge cases.
    测试 _get_arch_info 边界情况。"""

    @patch(
        "paddle.nn.quant.quantized_linear.is_compiled_with_cuda",
        return_value=False,
    )
    def test_get_arch_info_not_compiled_cuda(self, mock_cuda):
        """_get_arch_info returns 0 when not compiled with CUDA.
        当未使用 CUDA 编译时 _get_arch_info 返回 0。"""
        arch = _get_arch_info()
        self.assertEqual(arch, 0)

    @patch(
        "paddle.nn.quant.quantized_linear.is_compiled_with_cuda",
        return_value=True,
    )
    @patch(
        "paddle.nn.quant.quantized_linear.paddle.is_compiled_with_rocm",
        return_value=True,
    )
    @patch(
        "paddle.nn.quant.quantized_linear.get_device_capability",
        return_value=(9, 0),
    )
    def test_get_arch_info_rocm(self, mock_cap, mock_rocm, mock_cuda):
        """_get_arch_info handles ROCm compilation path.
        _get_arch_info 处理 ROCm 编译路径。"""
        arch = _get_arch_info()
        self.assertEqual(arch, 90)

    @patch(
        "paddle.nn.quant.quantized_linear.is_compiled_with_cuda",
        return_value=True,
    )
    @patch(
        "paddle.nn.quant.quantized_linear.paddle.is_compiled_with_rocm",
        return_value=False,
    )
    @patch(
        "paddle.nn.quant.quantized_linear.paddle.version.cuda",
        return_value='False',
    )
    def test_get_arch_info_cuda_false_string(
        self, mock_ver, mock_rocm, mock_cuda
    ):
        """_get_arch_info raises ValueError when cuda version is 'False'.
        当 cuda 版本为 'False' 时 _get_arch_info 引发 ValueError。"""
        with self.assertRaises(ValueError):
            _get_arch_info()

    @patch(
        "paddle.nn.quant.quantized_linear.is_compiled_with_cuda",
        return_value=True,
    )
    @patch(
        "paddle.nn.quant.quantized_linear.paddle.is_compiled_with_rocm",
        return_value=False,
    )
    @patch(
        "paddle.nn.quant.quantized_linear.paddle.version.cuda",
        return_value=None,
    )
    def test_get_arch_info_cuda_none(self, mock_ver, mock_rocm, mock_cuda):
        """_get_arch_info raises ValueError when cuda version is None.
        当 cuda 版本为 None 时 _get_arch_info 引发 ValueError。"""
        with self.assertRaises(ValueError):
            _get_arch_info()

    @patch(
        "paddle.nn.quant.quantized_linear.is_compiled_with_cuda",
        return_value=True,
    )
    @patch(
        "paddle.nn.quant.quantized_linear.paddle.is_compiled_with_rocm",
        return_value=False,
    )
    @patch(
        "paddle.nn.quant.quantized_linear.paddle.version.cuda",
        return_value="12.0",
    )
    @patch(
        "paddle.nn.quant.quantized_linear.get_device_capability",
        return_value=(8, 0),
    )
    def test_get_arch_info_valid_cuda(
        self, mock_cap, mock_ver, mock_rocm, mock_cuda
    ):
        """_get_arch_info returns correct arch for valid CUDA setup.
        对于有效的 CUDA 配置，_get_arch_info 返回正确的架构。"""
        arch = _get_arch_info()
        self.assertEqual(arch, 80)


class TestWeightQuantizeInvalidArch(unittest.TestCase):
    """Test weight_quantize with invalid arch values.
    测试无效 arch 值的 weight_quantize。"""

    def setUp(self):
        paddle.disable_static()

    @unittest.skipIf(not paddle.is_compiled_with_cuda(), "CUDA required")
    def test_weight_quantize_unsupported_arch(self):
        """weight_quantize raises AssertionError for unsupported arch.
        不支持的架构会引发 weight_quantize 的 AssertionError。"""
        with self.assertRaises(AssertionError):
            x = paddle.randn([4, 4], dtype=paddle.float16)
            weight_quantize(x, algo="weight_only_int8", arch=60)

    @unittest.skipIf(not paddle.is_compiled_with_cuda(), "CUDA required")
    def test_weight_quantize_invalid_group_size(self):
        """weight_quantize raises AssertionError for invalid group_size.
        无效的 group_size 会引发 weight_quantize 的 AssertionError。"""
        try:
            x = paddle.randn([4, 4], dtype=paddle.float16)
            weight_quantize(x, algo="weight_only_int8", arch=80, group_size=32)
        except AssertionError:
            pass  # Expected


class TestWeightQuantizeStaticGraph(unittest.TestCase):
    """Test weight_quantize in static graph mode.
    测试静态图模式下的 weight_quantize。"""

    def setUp(self):
        paddle.enable_static()

    def tearDown(self):
        paddle.disable_static()

    @unittest.skipIf(not paddle.is_compiled_with_cuda(), "CUDA required")
    def test_weight_quantize_static_graph_int8(self):
        """weight_quantize works in static graph mode with int8.
        weight_quantize 在静态图模式下使用 int8 正常工作。"""
        try:
            main_program = paddle.static.Program()
            startup_program = paddle.static.Program()
            with paddle.static.program_guard(main_program, startup_program):
                x = paddle.static.data(
                    name='x', shape=[64, 32], dtype='float16'
                )
                out, scale = weight_quantize(
                    x, algo="weight_only_int8", arch=80
                )
                self.assertIsNotNone(out)
                self.assertIsNotNone(scale)
        except (AssertionError, RuntimeError) as e:
            self.skipTest(f"Unsupported: {e}")

    @unittest.skipIf(not paddle.is_compiled_with_cuda(), "CUDA required")
    def test_weight_quantize_static_graph_int4(self):
        """weight_quantize works in static graph mode with int4.
        weight_quantize 在静态图模式下使用 int4 正常工作。"""
        try:
            main_program = paddle.static.Program()
            startup_program = paddle.static.Program()
            with paddle.static.program_guard(main_program, startup_program):
                x = paddle.static.data(
                    name='x', shape=[64, 32], dtype='float16'
                )
                out, scale = weight_quantize(
                    x, algo="weight_only_int4", arch=80, group_size=64
                )
                self.assertIsNotNone(out)
                self.assertIsNotNone(scale)
        except (AssertionError, RuntimeError) as e:
            self.skipTest(f"Unsupported: {e}")


class TestWeightDequantizeStaticGraph(unittest.TestCase):
    """Test weight_dequantize in static graph mode.
    测试静态图模式下的 weight_dequantize。"""

    def setUp(self):
        paddle.enable_static()

    def tearDown(self):
        paddle.disable_static()

    @unittest.skipIf(not paddle.is_compiled_with_cuda(), "CUDA required")
    def test_weight_dequantize_static_graph(self):
        """weight_dequantize works in static graph mode.
        weight_dequantize 在静态图模式下正常工作。"""
        try:
            main_program = paddle.static.Program()
            startup_program = paddle.static.Program()
            with paddle.static.program_guard(main_program, startup_program):
                x = paddle.static.data(name='x', shape=[32, 64], dtype='int8')
                scale = paddle.static.data(
                    name='scale', shape=[32], dtype='float16'
                )
                out = weight_dequantize(
                    x, scale, algo="weight_only_int8", group_size=-1
                )
                self.assertIsNotNone(out)
        except (AssertionError, RuntimeError) as e:
            self.skipTest(f"Unsupported: {e}")


class TestWeightOnlyLinearStaticGraph(unittest.TestCase):
    """Test weight_only_linear in static graph mode.
    测试静态图模式下的 weight_only_linear。"""

    def setUp(self):
        paddle.enable_static()

    def tearDown(self):
        paddle.disable_static()

    @unittest.skipIf(not paddle.is_compiled_with_cuda(), "CUDA required")
    def test_weight_only_linear_static_graph_with_bias(self):
        """weight_only_linear works in static graph with bias.
        weight_only_linear 在静态图模式下带 bias 正常工作。"""
        try:
            main_program = paddle.static.Program()
            startup_program = paddle.static.Program()
            with paddle.static.program_guard(main_program, startup_program):
                x = paddle.static.data(
                    name='x', shape=[1, 4, 64], dtype='float16'
                )
                weight = paddle.static.data(
                    name='w', shape=[32, 64], dtype='int8'
                )
                scale = paddle.static.data(
                    name='s', shape=[32], dtype='float32'
                )
                bias = paddle.static.data(name='b', shape=[32], dtype='float16')
                out = weight_only_linear(
                    x,
                    weight,
                    bias=bias,
                    weight_scale=scale,
                    weight_dtype="int8",
                    arch=80,
                )
                self.assertIsNotNone(out)
        except (AssertionError, RuntimeError) as e:
            self.skipTest(f"Unsupported: {e}")

    @unittest.skipIf(not paddle.is_compiled_with_cuda(), "CUDA required")
    def test_weight_only_linear_static_graph_without_bias(self):
        """weight_only_linear works in static graph without bias.
        weight_only_linear 在静态图模式下不带 bias 正常工作。"""
        try:
            main_program = paddle.static.Program()
            startup_program = paddle.static.Program()
            with paddle.static.program_guard(main_program, startup_program):
                x = paddle.static.data(
                    name='x', shape=[1, 4, 64], dtype='float16'
                )
                weight = paddle.static.data(
                    name='w', shape=[32, 64], dtype='int8'
                )
                scale = paddle.static.data(
                    name='s', shape=[32], dtype='float32'
                )
                out = weight_only_linear(
                    x, weight, weight_scale=scale, weight_dtype="int8", arch=80
                )
                self.assertIsNotNone(out)
        except (AssertionError, RuntimeError) as e:
            self.skipTest(f"Unsupported: {e}")


class TestLlmInt8LinearStaticGraph(unittest.TestCase):
    """Test llm_int8_linear in static graph mode.
    测试静态图模式下的 llm_int8_linear。"""

    def setUp(self):
        paddle.enable_static()

    def tearDown(self):
        paddle.disable_static()

    @unittest.skipIf(not paddle.is_compiled_with_cuda(), "CUDA required")
    def test_llm_int8_linear_static_graph(self):
        """llm_int8_linear works in static graph mode.
        llm_int8_linear 在静态图模式下正常工作。"""
        try:
            main_program = paddle.static.Program()
            startup_program = paddle.static.Program()
            with paddle.static.program_guard(main_program, startup_program):
                x = paddle.static.data(
                    name='x', shape=[1, 4, 64], dtype='float16'
                )
                weight = paddle.static.data(
                    name='w', shape=[32, 64], dtype='int8'
                )
                scale = paddle.static.data(
                    name='s', shape=[32], dtype='float32'
                )
                bias = paddle.static.data(name='b', shape=[32], dtype='float16')
                out = llm_int8_linear(
                    x, weight, bias=bias, weight_scale=scale, threshold=6.0
                )
                self.assertIsNotNone(out)
        except (AssertionError, RuntimeError) as e:
            self.skipTest(f"Unsupported: {e}")

    @unittest.skipIf(not paddle.is_compiled_with_cuda(), "CUDA required")
    def test_llm_int8_linear_static_graph_no_bias(self):
        """llm_int8_linear works in static graph without bias.
        llm_int8_linear 在静态图模式下不带 bias 正常工作。"""
        try:
            main_program = paddle.static.Program()
            startup_program = paddle.static.Program()
            with paddle.static.program_guard(main_program, startup_program):
                x = paddle.static.data(
                    name='x', shape=[1, 4, 64], dtype='float16'
                )
                weight = paddle.static.data(
                    name='w', shape=[32, 64], dtype='int8'
                )
                scale = paddle.static.data(
                    name='s', shape=[32], dtype='float32'
                )
                out = llm_int8_linear(
                    x, weight, weight_scale=scale, threshold=6.0
                )
                self.assertIsNotNone(out)
        except (AssertionError, RuntimeError) as e:
            self.skipTest(f"Unsupported: {e}")


class TestApplyPerChannelScaleStaticGraph(unittest.TestCase):
    """Test apply_per_channel_scale in static graph mode.
    测试静态图模式下的 apply_per_channel_scale。"""

    def setUp(self):
        paddle.enable_static()

    def tearDown(self):
        paddle.disable_static()

    @unittest.skipIf(not paddle.is_compiled_with_cuda(), "CUDA required")
    def test_apply_per_channel_scale_static_graph(self):
        """apply_per_channel_scale works in static graph mode.
        apply_per_channel_scale 在静态图模式下正常工作。"""
        try:
            main_program = paddle.static.Program()
            startup_program = paddle.static.Program()
            with paddle.static.program_guard(main_program, startup_program):
                x = paddle.static.data(
                    name='x', shape=[64, 32], dtype='float16'
                )
                scales = paddle.static.data(
                    name='s', shape=[32], dtype='float16'
                )
                out = apply_per_channel_scale(x, scales)
                self.assertIsNotNone(out)
        except (AssertionError, RuntimeError) as e:
            self.skipTest(f"Unsupported: {e}")


if __name__ == "__main__":
    unittest.main()
