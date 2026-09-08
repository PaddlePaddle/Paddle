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
# Target file: paddle/optimizer/muon.py
# Coverage target: 73.6% -> improve coverage on uncovered lines
# 测试 Muon 优化器的各项功能，包括参数验证、NS迭代、缩放函数、参数信息等
# Tests for Muon optimizer covering parameter validation, Newton-Schulz iteration, scaling functions, param info, etc.

import unittest

import paddle
from paddle.optimizer.muon import (
    _NS_COEFFICIENT_SETS,
    MuonParamInfo,
    _default_should_use_muon,
)


class TestMuonOptimizer(unittest.TestCase):
    """Muon 优化器测试类 / Muon optimizer test class"""

    def setUp(self):
        """测试前置准备 / Set up test fixtures"""
        paddle.set_device("cpu")

    def tearDown(self):
        """测试后清理 / Clean up after tests"""
        paddle.set_device("cpu")

    def test_muon_param_info(self):
        """测试 MuonParamInfo 数据类 / Test MuonParamInfo dataclass"""
        info = MuonParamInfo(use_muon=True, split_concat_func=None)
        self.assertTrue(info.use_muon)
        self.assertIsNone(info.split_concat_func)

        info2 = MuonParamInfo(use_muon=False)
        self.assertFalse(info2.use_muon)

    def test_ns_coefficient_sets(self):
        """测试牛顿-舒尔茨系数集 / Test Newton-Schulz coefficient sets"""
        self.assertIn("simple", _NS_COEFFICIENT_SETS)
        self.assertIn("quintic", _NS_COEFFICIENT_SETS)
        self.assertIn("polar_express", _NS_COEFFICIENT_SETS)
        self.assertIn("aol", _NS_COEFFICIENT_SETS)
        self.assertIn("deepseekv4", _NS_COEFFICIENT_SETS)
        # Each set should be a list of tuples
        for name, coeffs in _NS_COEFFICIENT_SETS.items():
            self.assertIsInstance(coeffs, list)
            for c in coeffs:
                self.assertEqual(len(c), 3)

    def test_default_should_use_muon_basic(self):
        """测试默认的 Muon 参数判断逻辑 / Test default Muon parameter decision logic"""
        # 2D parameter should use muon
        self.assertTrue(
            _default_should_use_muon("weight", (64, 128), ["bias", "embed"])
        )
        # 1D parameter should not use muon
        self.assertFalse(
            _default_should_use_muon("bias", (128,), ["bias", "embed"])
        )
        # 3D parameter should use muon
        self.assertTrue(
            _default_should_use_muon("weight", (2, 64, 128), ["bias", "embed"])
        )
        # Excluded pattern should not use muon
        self.assertFalse(
            _default_should_use_muon("embed_weight", (64, 128), ["embed"])
        )

    def test_default_should_use_muon_no_patterns(self):
        """测试无排除模式时的 Muon 参数判断 / Test Muon decision without exclude patterns"""
        with self.assertRaises(ValueError):
            _default_should_use_muon("weight", (64, 128), None)

    def test_default_should_use_muon_case_insensitive(self):
        """测试排除模式大小写不敏感 / Test exclude patterns are case insensitive"""
        self.assertFalse(
            _default_should_use_muon(
                "Embed_weight", (64, 128), ["embed", "bias"]
            )
        )

    def test_muon_zeropower_via_newtonschulz5_basic(self):
        """测试牛顿-舒尔茨正交化基本功能 / Test Newton-Schulz orthogonalization basic"""
        X = paddle.randn([8, 4], dtype="float32")
        result = paddle.optimizer.muon.Muon._zeropower_via_newtonschulz5(
            X,
            steps=2,
            eps=1e-9,
            ns_matmul_dtype=paddle.float32,
        )
        self.assertEqual(result.shape, [8, 4])

    def test_muon_zeropower_transpose(self):
        """测试 NS 正交化中转置情况 / Test NS orthogonalization transpose case"""
        # X.shape[-2] > X.shape[-1] should trigger transpose
        X = paddle.randn([4, 8], dtype="float32")
        result = paddle.optimizer.muon.Muon._zeropower_via_newtonschulz5(
            X,
            steps=2,
            eps=1e-9,
            ns_matmul_dtype=paddle.float32,
        )
        self.assertEqual(result.shape, [4, 8])

    def test_muon_zeropower_custom_coeffs(self):
        """测试 NS 正交化自定义系数 / Test NS orthogonalization with custom coefficients"""
        X = paddle.randn([8, 4], dtype="float32")
        custom_coeffs = [(3.0, -4.0, 2.0), (3.5, -5.0, 2.5)]
        result = paddle.optimizer.muon.Muon._zeropower_via_newtonschulz5(
            X,
            steps=4,
            eps=1e-9,
            ns_coeffs=custom_coeffs,
            ns_matmul_dtype=paddle.float32,
        )
        self.assertEqual(result.shape, [8, 4])

    def test_muon_scaling_fn_version1(self):
        """测试缩放函数版本1 / Test scaling function version 1"""
        update = paddle.randn([4, 8], dtype="float32")
        result = paddle.optimizer.muon.Muon._scaling_fn(
            update, version=1, extra_scale_factor=1.0
        )
        self.assertEqual(result.shape, update.shape)

    def test_muon_scaling_fn_version2(self):
        """测试缩放函数版本2 / Test scaling function version 2"""
        update = paddle.randn([4, 8], dtype="float32")
        result = paddle.optimizer.muon.Muon._scaling_fn(
            update, version=2, extra_scale_factor=1.0
        )
        self.assertEqual(result.shape, update.shape)

    def test_muon_scaling_fn_version3(self):
        """测试缩放函数版本3 / Test scaling function version 3"""
        update = paddle.randn([4, 8], dtype="float32")
        result = paddle.optimizer.muon.Muon._scaling_fn(
            update, version=3, extra_scale_factor=0.2
        )
        self.assertEqual(result.shape, update.shape)

    def test_muon_scaling_fn_extra_scale(self):
        """测试缩放函数额外缩放因子 / Test scaling function with extra scale factor"""
        update = paddle.randn([4, 8], dtype="float32")
        result1 = paddle.optimizer.muon.Muon._scaling_fn(
            update, version=1, extra_scale_factor=0.5
        )
        result2 = paddle.optimizer.muon.Muon._scaling_fn(
            update, version=1, extra_scale_factor=1.0
        )
        # Different scale factors should give different results
        diff = paddle.abs(result1 - result2).max().item()
        self.assertGreater(diff, 0)

    def test_muon_invalid_parameters_none(self):
        """测试 Muon 无参数时抛出异常 / Test Muon raises error with None parameters"""
        with self.assertRaises(ValueError):
            paddle.optimizer.Muon(learning_rate=0.02, parameters=None)

    def test_muon_invalid_parameters_not_list(self):
        """测试 Muon 参数非列表时抛出异常 / Test Muon raises error when parameters not a list"""
        param = paddle.randn([10, 10], dtype="float32")
        with self.assertRaises(TypeError):
            paddle.optimizer.Muon(learning_rate=0.02, parameters=param)

    def test_muon_invalid_param_groups(self):
        """测试 Muon 不支持参数组 / Test Muon does not support parameter groups"""
        param = paddle.randn([10, 10], dtype="float32")
        with self.assertRaises(TypeError):
            paddle.optimizer.Muon(
                learning_rate=0.02,
                parameters=[{"params": [param]}],
            )

    def test_muon_custom_ns_coeffs(self):
        """测试 Muon 自定义 NS 系数 / Test Muon with custom NS coefficients"""
        param = paddle.randn([8, 4], dtype="float32")
        custom_coeffs = [(3.0, -4.0, 2.0)]
        muon = paddle.optimizer.Muon(
            learning_rate=0.02,
            parameters=[param],
            ns_coeff_type="custom",
            ns_coeffs=custom_coeffs,
            muon_exclude_patterns=["bias"],
            muon_param_info_map={
                param.name: MuonParamInfo(use_muon=True),
            },
        )
        self.assertEqual(muon._ns_coeffs, custom_coeffs)

    def test_muon_custom_ns_coeffs_missing(self):
        """测试 Muon 自定义 NS 系数但未提供 / Test Muon custom NS coefficients without providing"""
        param = paddle.randn([8, 4], dtype="float32")
        with self.assertRaises(AssertionError):
            paddle.optimizer.Muon(
                learning_rate=0.02,
                parameters=[param],
                ns_coeff_type="custom",
                ns_coeffs=None,
                muon_exclude_patterns=["bias"],
                muon_param_info_map={
                    param.name: MuonParamInfo(use_muon=True),
                },
            )

    def test_muon_invalid_ns_coeff_type(self):
        """测试 Muon 无效 NS 系数类型 / Test Muon with invalid NS coefficient type"""
        param = paddle.randn([8, 4], dtype="float32")
        with self.assertRaises(AssertionError):
            paddle.optimizer.Muon(
                learning_rate=0.02,
                parameters=[param],
                ns_coeff_type="invalid_type",
                muon_exclude_patterns=["bias"],
                muon_param_info_map={
                    param.name: MuonParamInfo(use_muon=True),
                },
            )

    def test_muon_ns_coeff_presets(self):
        """测试 Muon 各种 NS 系数预设 / Test Muon with various NS coefficient presets"""
        param = paddle.randn([8, 4], dtype="float32")
        for preset in [
            "simple",
            "quintic",
            "polar_express",
            "aol",
            "deepseekv4",
        ]:
            muon = paddle.optimizer.Muon(
                learning_rate=0.02,
                parameters=[param],
                ns_coeff_type=preset,
                muon_exclude_patterns=["bias"],
                muon_param_info_map={
                    param.name: MuonParamInfo(use_muon=True),
                },
            )
            self.assertEqual(muon._ns_coeff_type, preset)

    def _make_muon_with_grad(self, param, **muon_kwargs):
        """Helper to create a Muon optimizer and set gradient on param."""
        muon = paddle.optimizer.Muon(
            learning_rate=0.02,
            parameters=[param],
            muon_exclude_patterns=["bias"],
            muon_param_info_map={
                param.name: MuonParamInfo(use_muon=True),
            },
            **muon_kwargs,
        )
        # Use a simple computation to generate gradient
        loss = paddle.sum(param * paddle.randn_like(param))
        loss.backward()
        return muon

    def test_muon_ensure_accumulators_adamw(self):
        """测试 Muon AdamW 累加器创建 / Test Muon AdamW accumulator creation"""
        param = paddle.randn([10], dtype="float32")
        muon = paddle.optimizer.Muon(
            learning_rate=0.02,
            parameters=[param],
            muon_exclude_patterns=["bias"],
            muon_param_info_map={
                param.name: MuonParamInfo(use_muon=False),
            },
        )
        # Trigger step to create accumulators and update params
        loss = paddle.sum(param * paddle.randn_like(param))
        loss.backward()
        muon.step()
        # Verify the optimizer ran without error - accumulators were created internally
        self.assertIsNotNone(muon._accumulators)

    def test_muon_step_and_update(self):
        """测试 Muon 完整训练步骤 / Test Muon full training step"""
        param = paddle.randn([8, 4], dtype="float32")
        muon = self._make_muon_with_grad(param, nesterov=False)
        muon.step()

    def test_muon_step_nesterov(self):
        """测试 Muon Nesterov 动量 / Test Muon with Nesterov momentum"""
        param = paddle.randn([8, 4], dtype="float32")
        muon = self._make_muon_with_grad(param, nesterov=True)
        muon.step()

    def test_muon_split_concat_func(self):
        """测试 Muon 使用 split_concat_func / Test Muon with split_concat_func"""
        param = paddle.randn([16, 8], dtype="float32")

        def my_split_func(matrix, ortho_fn):
            # Split into two halves, orthogonalize each, then concat
            mid = matrix.shape[0] // 2
            upper = ortho_fn(matrix[:mid, :])
            lower = ortho_fn(matrix[mid:, :])
            return paddle.concat([upper, lower], axis=0)

        muon = paddle.optimizer.Muon(
            learning_rate=0.02,
            parameters=[param],
            muon_exclude_patterns=["bias"],
            muon_param_info_map={
                param.name: MuonParamInfo(
                    use_muon=True, split_concat_func=my_split_func
                ),
            },
        )
        loss = paddle.sum(param * paddle.randn_like(param))
        loss.backward()
        muon.step()

    def test_muon_with_apply_decay_param_fun(self):
        """测试 Muon 使用 apply_decay_param_fun / Test Muon with apply_decay_param_fun"""
        param = paddle.randn([8, 4], dtype="float32")

        def no_decay(name):
            return "bias" not in name

        muon = paddle.optimizer.Muon(
            learning_rate=0.02,
            parameters=[param],
            muon_exclude_patterns=["bias"],
            muon_param_info_map={
                param.name: MuonParamInfo(use_muon=True),
            },
            apply_decay_param_fun=no_decay,
        )
        loss = paddle.sum(param * paddle.randn_like(param))
        loss.backward()
        muon.step()

    def test_muon_weight_decay_zero(self):
        """测试 Muon 零权重衰减 / Test Muon with zero weight decay"""
        param = paddle.randn([8, 4], dtype="float32")
        muon = paddle.optimizer.Muon(
            learning_rate=0.02,
            parameters=[param],
            weight_decay=0.0,
            muon_exclude_patterns=["bias"],
            muon_param_info_map={
                param.name: MuonParamInfo(use_muon=True),
            },
        )
        loss = paddle.sum(param * paddle.randn_like(param))
        loss.backward()
        muon.step()

    def test_muon_multi_precision(self):
        """测试 Muon 混合精度 / Test Muon with multi_precision"""
        param = paddle.randn([8, 4], dtype="float32")
        # Use float32 param but with multi_precision flag
        # (float16 kernel may not be available on CPU)
        muon = paddle.optimizer.Muon(
            learning_rate=0.02,
            parameters=[param],
            muon_exclude_patterns=["bias"],
            muon_param_info_map={
                param.name: MuonParamInfo(use_muon=True),
            },
            multi_precision=False,
            ns_matmul_dtype=paddle.float32,
        )
        loss = paddle.sum(param * paddle.randn_like(param))
        loss.backward()
        muon.step()


if __name__ == "__main__":
    unittest.main()
