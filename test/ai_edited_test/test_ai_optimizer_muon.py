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

# [AUTO-GENERATED] Test file for paddle.optimizer.muon
# 覆盖模块: paddle/optimizer/muon.py
# 未覆盖行: 135,141,232,236,238,245,307,308,344,345,346,347,348,349,351,356,414,415,417,546,547,548,549,589,602,609,657,662,684,685
# Covered module: paddle/optimizer/muon.py
# Uncovered lines: 135,141,232,236,238,245,307,308,344,345,346,347,348,349,351,356,414,415,417,546,547,548,549,589,602,609,657,662,684,685

import unittest

import numpy as np

import paddle
from paddle.optimizer.muon import (
    _NS_COEFFICIENT_SETS,
    Muon,
    MuonParamInfo,
    _default_should_use_muon,
)


class TestDefaultShouldUseMuon(unittest.TestCase):
    """测试 _default_should_use_muon 函数
    Test _default_should_use_muon function"""

    def test_should_use_muon_2d(self):
        """测试2D参数应该使用 Muon
        Test 2D parameter should use Muon"""
        result = _default_should_use_muon("weight", (128, 64), [])
        self.assertTrue(result)

    def test_should_use_muon_3d(self):
        """测试3D参数应该使用 Muon
        Test 3D parameter should use Muon"""
        result = _default_should_use_muon("weight", (4, 128, 64), [])
        self.assertTrue(result)

    def test_should_not_use_muon_1d(self):
        """测试1D参数不应使用 Muon
        Test 1D parameter should not use Muon"""
        result = _default_should_use_muon("bias", (128,), [])
        self.assertFalse(result)

    def test_should_not_use_muon_4d(self):
        """测试4D参数不应使用 Muon
        Test 4D parameter should not use Muon"""
        result = _default_should_use_muon("conv_weight", (3, 3, 3, 3), [])
        self.assertFalse(result)

    def test_should_not_use_muon_exclude_pattern(self):
        """测试匹配排除模式的参数不应使用 Muon
        Test parameter matching exclude pattern should not use Muon"""
        result = _default_should_use_muon("embed_weight", (128, 64), ['embed'])
        self.assertFalse(result)

    def test_should_not_use_muon_bias(self):
        """测试 bias 参数不应使用 Muon
        Test bias parameter should not use Muon"""
        result = _default_should_use_muon("linear.bias", (128, 64), ['bias'])
        self.assertFalse(result)

    def test_should_use_muon_none_patterns(self):
        """测试 exclude_patterns 为 None 时抛出 ValueError
        Test ValueError when exclude_patterns is None"""
        with self.assertRaises(ValueError):
            _default_should_use_muon("weight", (128, 64), None)

    def test_should_use_muon_case_insensitive(self):
        """测试排除模式匹配不区分大小写
        Test exclude pattern matching is case-insensitive"""
        result = _default_should_use_muon("EMBED_weight", (128, 64), ['embed'])
        self.assertFalse(result)


class TestNSCoefficientSets(unittest.TestCase):
    """测试 Newton-Schulz 系数集
    Test Newton-Schulz coefficient sets"""

    def test_simple_coefficients(self):
        """测试 simple 系数集
        Test simple coefficient set"""
        self.assertIn("simple", _NS_COEFFICIENT_SETS)
        self.assertEqual(len(_NS_COEFFICIENT_SETS["simple"]), 1)

    def test_quintic_coefficients(self):
        """测试 quintic 系数集
        Test quintic coefficient set"""
        self.assertIn("quintic", _NS_COEFFICIENT_SETS)
        self.assertEqual(len(_NS_COEFFICIENT_SETS["quintic"]), 5)

    def test_polar_express_coefficients(self):
        """测试 polar_express 系数集
        Test polar_express coefficient set"""
        self.assertIn("polar_express", _NS_COEFFICIENT_SETS)
        self.assertEqual(len(_NS_COEFFICIENT_SETS["polar_express"]), 8)

    def test_aol_coefficients(self):
        """测试 aol 系数集
        Test aol coefficient set"""
        self.assertIn("aol", _NS_COEFFICIENT_SETS)
        self.assertEqual(len(_NS_COEFFICIENT_SETS["aol"]), 4)

    def test_deepseekv4_coefficients(self):
        """测试 deepseekv4 系数集
        Test deepseekv4 coefficient set"""
        self.assertIn("deepseekv4", _NS_COEFFICIENT_SETS)
        self.assertEqual(len(_NS_COEFFICIENT_SETS["deepseekv4"]), 10)


class TestMuonParamInfo(unittest.TestCase):
    """测试 MuonParamInfo 数据类
    Test MuonParamInfo dataclass"""

    def test_default_values(self):
        """测试 MuonParamInfo 默认值
        Test MuonParamInfo default values"""
        info = MuonParamInfo()
        self.assertTrue(info.use_muon)
        self.assertIsNone(info.split_concat_func)

    def test_custom_values(self):
        """测试 MuonParamInfo 自定义值
        Test MuonParamInfo with custom values"""

        def split_fn(matrix, ortho_fn, **kwargs):
            return ortho_fn(matrix)

        info = MuonParamInfo(use_muon=False, split_concat_func=split_fn)
        self.assertFalse(info.use_muon)
        self.assertEqual(info.split_concat_func, split_fn)


class TestMuonInit(unittest.TestCase):
    """测试 Muon 优化器初始化
    Test Muon optimizer initialization"""

    def test_muon_none_parameters(self):
        """测试 parameters=None 时抛出 ValueError
        Test ValueError when parameters is None"""
        with self.assertRaises(ValueError):
            Muon(parameters=None)

    def test_muon_dict_parameters(self):
        """测试字典参数列表时抛出 TypeError
        Test TypeError when parameters is a list of dicts"""
        linear = paddle.nn.Linear(10, 10)
        with self.assertRaises(TypeError):
            Muon(parameters=[{'params': linear.parameters()}])

    def test_muon_non_list_parameters(self):
        """测试非列表参数时抛出 TypeError
        Test TypeError when parameters is not a list"""
        with self.assertRaises(TypeError):
            Muon(parameters="invalid")

    def test_muon_invalid_grad_clip(self):
        """测试无效 grad_clip 时抛出 TypeError
        Test TypeError with invalid grad_clip"""
        linear = paddle.nn.Linear(10, 10)
        with self.assertRaises(TypeError):
            Muon(
                parameters=linear.parameters(),
                grad_clip="invalid_clip",
            )


class TestMuonScalingFn(unittest.TestCase):
    """测试 Muon._scaling_fn 静态方法
    Test Muon._scaling_fn static method"""

    def test_scaling_fn_version1(self):
        """测试 version=1 的 scaling 函数
        Test scaling function with version=1"""
        tensor = paddle.randn([8, 4])
        result = Muon._scaling_fn(tensor, version=1, extra_scale_factor=0.2)
        # version 1: scale = max(1, dout/din)^0.5 * extra_scale_factor
        expected_scale = max(1, 4 / 8) ** 0.5 * 0.2
        np.testing.assert_allclose(
            result.numpy(),
            tensor.numpy() * expected_scale,
            rtol=1e-5,
            atol=1e-5,
        )

    def test_scaling_fn_version2(self):
        """测试 version=2 的 scaling 函数
        Test scaling function with version=2"""
        tensor = paddle.randn([4, 8])
        result = Muon._scaling_fn(tensor, version=2, extra_scale_factor=1.0)
        # version 2: scale = (dout/din)^0.5 * extra_scale_factor
        expected_scale = (8 / 4) ** 0.5 * 1.0
        np.testing.assert_allclose(
            result.numpy(),
            tensor.numpy() * expected_scale,
            rtol=1e-5,
            atol=1e-5,
        )

    def test_scaling_fn_version3(self):
        """测试 version=3 的 scaling 函数
        Test scaling function with version=3"""
        tensor = paddle.randn([4, 8])
        result = Muon._scaling_fn(tensor, version=3, extra_scale_factor=0.2)
        # version 3: scale = max(dout, din)^0.5 * extra_scale_factor
        expected_scale = max(8, 4) ** 0.5 * 0.2
        np.testing.assert_allclose(
            result.numpy(),
            tensor.numpy() * expected_scale,
            rtol=1e-5,
            atol=1e-5,
        )


class TestZeropowerNewtonschulz5(unittest.TestCase):
    """测试 _zeropower_via_newtonschulz5 静态方法
    Test _zeropower_via_newtonschulz5 static method"""

    def test_basic_orthogonalization(self):
        """测试基本的正交化
        Test basic orthogonalization"""
        x = paddle.randn([4, 4], dtype='float32')
        result = Muon._zeropower_via_newtonschulz5(
            x, steps=3, ns_matmul_dtype=paddle.float32
        )
        self.assertEqual(result.shape, [4, 4])

    def test_tall_matrix(self):
        """测试高矩阵 (rows > cols) 时的转置处理
        Test tall matrix (rows > cols) transpose handling"""
        x = paddle.randn([8, 4], dtype='float32')
        result = Muon._zeropower_via_newtonschulz5(
            x, steps=3, ns_matmul_dtype=paddle.float32
        )
        self.assertEqual(result.shape, [8, 4])

    def test_wide_matrix(self):
        """测试宽矩阵 (rows < cols)
        Test wide matrix (rows < cols)"""
        x = paddle.randn([4, 8], dtype='float32')
        result = Muon._zeropower_via_newtonschulz5(
            x, steps=3, ns_matmul_dtype=paddle.float32
        )
        self.assertEqual(result.shape, [4, 8])

    def test_custom_coeffs(self):
        """测试使用自定义系数
        Test with custom coefficients"""
        x = paddle.randn([4, 4], dtype='float32')
        custom_coeffs = [(3.4445, -4.7750, 2.0315)]
        result = Muon._zeropower_via_newtonschulz5(
            x, steps=1, ns_coeffs=custom_coeffs, ns_matmul_dtype=paddle.float32
        )
        self.assertEqual(result.shape, [4, 4])

    def test_zero_steps(self):
        """测试0步迭代 (仅归一化)
        Test zero-step iteration (normalization only)"""
        x = paddle.randn([4, 4], dtype='float32')
        result = Muon._zeropower_via_newtonschulz5(
            x, steps=0, ns_matmul_dtype=paddle.float32
        )
        self.assertEqual(result.shape, [4, 4])


if __name__ == '__main__':
    unittest.main()
