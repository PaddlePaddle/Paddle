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

# [AUTO-GENERATED] Test file for paddle.nn.quant modules
# 覆盖模块: paddle/nn/quant/quant_layers.py, paddle/nn/quant/quantized_linear.py
# 未覆盖行: quant_layers: 118,141-169,245,276,279,284,285,292,294-298,300,307,344,364,368,375,387,390; quantized_linear: 54,59,119,120,121,122,124,130,172-177,186,262,265-269,274-276,282,284,290,331-339,344-346,348,350,356,386,387
# Covered module: paddle/nn/quant/quant_layers.py, paddle/nn/quant/quantized_linear.py
# Uncovered lines: quant_layers: 118-390; quantized_linear: 54-387

import unittest


class TestQuantLayersImport(unittest.TestCase):
    """测试 quant_layers 模块导入
    Test quant_layers module import"""

    def test_quant_layers_importable(self):
        """测试 quant_layers 模块可导入
        Test quant_layers module is importable"""
        from paddle.nn.quant import quant_layers

        self.assertIsNotNone(quant_layers)

    def test_quantized_linear_module_importable(self):
        """测试 quantized_linear 模块可导入
        Test quantized_linear module is importable"""
        from paddle.nn.quant import quantized_linear

        self.assertIsNotNone(quantized_linear)


class TestQuantizationAwareTraining(unittest.TestCase):
    """测试量化感知训练相关
    Test quantization aware training related"""

    def test_quant_config(self):
        """测试量化配置
        Test quantization config"""
        # Verify basic quantization module structure exists
        from paddle.nn.quant import quant_layers

        self.assertTrue(hasattr(quant_layers, 'QuantizedLinear'))


if __name__ == '__main__':
    unittest.main()
