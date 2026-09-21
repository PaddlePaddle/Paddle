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

import os
import sys
import types
import unittest
from unittest import mock

import numpy as np

import paddle
from paddle.optimizer import muon as muon_module
from paddle.optimizer.muon import (
    _NS_COEFFICIENT_SETS,
    Muon,
    MuonParamInfo,
    _default_should_use_muon,
    _load_symmetric_gemm,
    _symmetric_gemm_is_profitable,
)

# Muon's default profitability thresholds, i.e. the defaults of
# ``symmetric_gemm_min_short_edge`` and ``symmetric_gemm_min_step_flops``.
# ``TestMuonSymmetricGemmInit.test_thresholds_default_to_documented_values``
# pins these against the constructor so the two cannot drift apart.
SYRK_DEFAULTS = (1024, 2e11)


def fake_gemm_symmetric(A, B, C=None, alpha=1.0, beta=0.0):
    """Dense stand-in for quack's ``gemm_symmetric``.

    Mirrors the public contract ``beta * C + alpha * (A @ B)`` with plain
    matmuls, so the Newton-Schulz plumbing can be tested without a GPU or a
    quack install.
    """
    out = alpha * paddle.matmul(A, B)
    if C is not None:
        out = out + beta * C
    return out


def quack_symmetric_gemm_blocker():
    """Why the real quack kernel cannot run here, or None if it can.

    Returning the reason instead of a bool keeps the skip auditable: a
    pipeline that is *supposed* to cover the production path can tell
    "no GPU in this job" apart from "quack failed to import".
    """
    if not paddle.is_compiled_with_cuda():
        return "paddle is not compiled with CUDA"
    cap = paddle.device.cuda.get_device_capability()
    if cap[0] not in (9, 10, 11):
        return f"compute capability {cap[0]}.{cap[1]} has no CuTe SYRK kernel"
    try:
        _load_symmetric_gemm()
    except RuntimeError as e:
        return str(e)
    return None


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


class TestSymmetricGemmProfitability(unittest.TestCase):
    """测试 _symmetric_gemm_is_profitable 门控
    Test the _symmetric_gemm_is_profitable gate"""

    def test_supported_dtypes(self):
        """测试 bf16/fp16 通过门控，fp32/fp64 被拒绝
        Test bf16/fp16 pass the gate while fp32/fp64 are rejected"""
        shape = [16, 2048, 2048]
        for dtype, expected in (
            (paddle.bfloat16, True),
            (paddle.float16, True),
            (paddle.float32, False),
            (paddle.float64, False),
        ):
            self.assertEqual(
                _symmetric_gemm_is_profitable(shape, dtype, *SYRK_DEFAULTS),
                expected,
                msg=str(dtype),
            )

    def test_large_batched_shapes_are_profitable(self):
        """测试大 batch 形状通过门控
        Test large batched shapes clear the gate"""
        for shape in ([16, 4096, 8192], [64, 2048, 2048], [16, 1024, 4096]):
            self.assertTrue(
                _symmetric_gemm_is_profitable(
                    shape, paddle.bfloat16, *SYRK_DEFAULTS
                ),
                msg=str(shape),
            )

    def test_large_2d_shape_is_profitable(self):
        """测试无 batch 维的大矩阵通过门控
        Test an unbatched large matrix clears the gate"""
        self.assertTrue(
            _symmetric_gemm_is_profitable(
                [4096, 8192], paddle.bfloat16, *SYRK_DEFAULTS
            )
        )

    def test_short_edge_is_the_row_dimension(self):
        """测试短边为行维时同样按短边判定
        Test the gate uses the short edge even when it is the row dim"""
        self.assertTrue(
            _symmetric_gemm_is_profitable(
                [16, 8192, 2048], paddle.bfloat16, *SYRK_DEFAULTS
            )
        )
        self.assertFalse(
            _symmetric_gemm_is_profitable(
                [16, 4096, 512], paddle.bfloat16, *SYRK_DEFAULTS
            )
        )

    def test_short_edge_below_minimum(self):
        """测试短边小于阈值时不走 SYRK
        Test a short edge below the threshold stays on cuBLAS"""
        for shape in ([111, 512, 4096], [16, 512, 1024], [6, 64, 1024]):
            self.assertFalse(
                _symmetric_gemm_is_profitable(
                    shape, paddle.bfloat16, *SYRK_DEFAULTS
                ),
                msg=str(shape),
            )

    def test_short_edge_not_aligned(self):
        """测试短边非 8 元素对齐时不走 SYRK
        Test an unaligned short edge stays on cuBLAS"""
        self.assertFalse(
            _symmetric_gemm_is_profitable(
                [16, 1036, 8192], paddle.bfloat16, *SYRK_DEFAULTS
            )
        )

    def test_long_edge_not_aligned(self):
        """测试长边（K 维）非 8 元素对齐时不走 SYRK
        Test an unaligned long edge (the K dim) stays on cuBLAS

        The long edge is the contiguous dimension of ``X @ X.T``, so quack
        raises ``Invalid mA.strides[0]`` on it just like it does on the output
        for an unaligned short edge.
        """
        for shape in ([4096, 4097], [4096, 4100], [16, 2048, 4097]):
            self.assertFalse(
                _symmetric_gemm_is_profitable(
                    shape, paddle.bfloat16, *SYRK_DEFAULTS
                ),
                msg=str(shape),
            )
        # Same fragment with the long edge padded back to alignment passes.
        self.assertTrue(
            _symmetric_gemm_is_profitable(
                [4096, 4104], paddle.bfloat16, *SYRK_DEFAULTS
            )
        )

    def test_flops_below_threshold(self):
        """测试单次调用 FLOPs 不足时不走 SYRK
        Test too few per-call FLOPs stays on cuBLAS"""
        self.assertFalse(
            _symmetric_gemm_is_profitable(
                [1, 1024, 1024], paddle.bfloat16, *SYRK_DEFAULTS
            )
        )
        self.assertFalse(
            _symmetric_gemm_is_profitable(
                [1024, 4096], paddle.bfloat16, *SYRK_DEFAULTS
            )
        )

    def test_thresholds_are_caller_supplied(self):
        """测试门控完全由调用方传入的阈值决定
        Test the gate is driven entirely by the caller's thresholds

        There is no module-level default: the same shape flips verdict purely
        on the thresholds handed in, which is what makes them configurable
        from ``Muon.__init__``.
        """
        shape = [1024, 4096]  # rejected under Muon's defaults
        self.assertFalse(
            _symmetric_gemm_is_profitable(
                shape, paddle.bfloat16, *SYRK_DEFAULTS
            )
        )
        self.assertTrue(
            _symmetric_gemm_is_profitable(shape, paddle.bfloat16, 512, 1)
        )
        # Raising either threshold alone rejects an otherwise passing shape.
        shape = [4096, 8192]
        self.assertTrue(
            _symmetric_gemm_is_profitable(
                shape, paddle.bfloat16, *SYRK_DEFAULTS
            )
        )
        self.assertFalse(
            _symmetric_gemm_is_profitable(shape, paddle.bfloat16, 8192, 2e11)
        )
        self.assertFalse(
            _symmetric_gemm_is_profitable(shape, paddle.bfloat16, 1024, 1e30)
        )

    def test_alignment_is_not_configurable(self):
        """测试对齐要求不受阈值配置影响
        Test the alignment requirement is unaffected by threshold overrides

        Alignment is a hard kernel constraint, not a tuning knob, so lowering
        the thresholds must not let an unaligned fragment through.
        """
        self.assertFalse(
            _symmetric_gemm_is_profitable([1036, 8192], paddle.bfloat16, 8, 1)
        )


class TestLoadSymmetricGemm(unittest.TestCase):
    """测试 _load_symmetric_gemm 的导入与缓存
    Test _load_symmetric_gemm import handling and caching"""

    def setUp(self):
        _load_symmetric_gemm.cache_clear()

    def tearDown(self):
        _load_symmetric_gemm.cache_clear()

    def test_import_failure_raises_runtime_error(self):
        """测试 quack 不可导入时抛出带指引的 RuntimeError
        Test an actionable RuntimeError when quack cannot be imported"""
        blocked = {"quack": None, "quack.gemm_interface": None}
        with (
            mock.patch.dict(sys.modules, blocked),
            self.assertRaises(RuntimeError) as ctx,
        ):
            _load_symmetric_gemm()
        self.assertIn("quack-kernels", str(ctx.exception))
        self.assertIsNotNone(ctx.exception.__cause__)

    def test_import_success_is_cached(self):
        """测试导入成功后结果被缓存，不再重复导入
        Test a successful lookup is cached instead of re-imported"""
        module = types.ModuleType("quack.gemm_interface")
        module.gemm_symmetric = fake_gemm_symmetric
        injected = {
            "quack": types.ModuleType("quack"),
            "quack.gemm_interface": module,
        }
        with mock.patch.dict(sys.modules, injected):
            self.assertIs(_load_symmetric_gemm(), fake_gemm_symmetric)
        # quack is gone from sys.modules again, the cache must still serve it.
        self.assertIs(_load_symmetric_gemm(), fake_gemm_symmetric)
        self.assertEqual(_load_symmetric_gemm.cache_info().currsize, 1)


class TestMuonSymmetricGemmInit(unittest.TestCase):
    """测试 use_symmetric_gemm 的构造期校验
    Test constructor validation of use_symmetric_gemm"""

    def _param(self):
        p = paddle.create_parameter(shape=[8, 8], dtype='float32')
        p.stop_gradient = False
        return p

    def test_disabled_by_default(self):
        """测试默认不开启对称 GEMM
        Test the symmetric GEMM path is off by default"""
        opt = Muon(parameters=[self._param()])
        self.assertFalse(opt._use_symmetric_gemm)

    def test_rejects_float32_ns_matmul_dtype(self):
        """测试 ns_matmul_dtype=float32 时被拒绝
        Test ns_matmul_dtype=float32 is rejected"""
        with self.assertRaises(ValueError) as ctx:
            Muon(
                parameters=[self._param()],
                use_symmetric_gemm=True,
                ns_matmul_dtype=paddle.float32,
            )
        self.assertIn("bfloat16 or float16", str(ctx.exception))

    def test_rejects_unsupported_capability(self):
        """测试算力不支持的 GPU 被拒绝
        Test a GPU without a CuTe symmetric kernel is rejected"""
        with (
            mock.patch.object(
                paddle, "is_compiled_with_cuda", return_value=True
            ),
            mock.patch.object(
                paddle.device.cuda, "get_device_capability", return_value=(8, 0)
            ),
            self.assertRaises(ValueError) as ctx,
        ):
            Muon(
                parameters=[self._param()],
                use_symmetric_gemm=True,
                ns_matmul_dtype=paddle.bfloat16,
            )
        self.assertIn("compute capability", str(ctx.exception))

    def test_rejects_cpu_only_build(self):
        """测试非 CUDA 编译时被拒绝
        Test a build without CUDA is rejected"""
        with (
            mock.patch.object(
                paddle, "is_compiled_with_cuda", return_value=False
            ),
            self.assertRaises(ValueError) as ctx,
        ):
            Muon(
                parameters=[self._param()],
                use_symmetric_gemm=True,
                ns_matmul_dtype=paddle.bfloat16,
            )
        self.assertIn("0.0", str(ctx.exception))

    def test_enabled_loads_kernel_eagerly(self):
        """测试条件满足时开启，并在构造期预加载 kernel
        Test the kernel is loaded eagerly once the config is valid"""
        loader = mock.Mock(return_value=fake_gemm_symmetric)
        with (
            mock.patch.object(
                paddle, "is_compiled_with_cuda", return_value=True
            ),
            mock.patch.object(
                paddle.device.cuda, "get_device_capability", return_value=(9, 0)
            ),
            mock.patch.object(muon_module, "_load_symmetric_gemm", loader),
        ):
            opt = Muon(
                parameters=[self._param()],
                use_symmetric_gemm=True,
                ns_matmul_dtype=paddle.bfloat16,
            )
        self.assertTrue(opt._use_symmetric_gemm)
        loader.assert_called_once_with()

    def test_thresholds_default_to_documented_values(self):
        """测试阈值默认值与文档一致
        Test the threshold defaults match the documented values

        The gate tests drive ``_symmetric_gemm_is_profitable`` with
        ``SYRK_DEFAULTS``; this is what keeps that tuple honest.
        """
        opt = Muon(parameters=[self._param()])
        self.assertEqual(
            (
                opt._symmetric_gemm_min_short_edge,
                opt._symmetric_gemm_min_step_flops,
            ),
            SYRK_DEFAULTS,
        )

    def test_thresholds_are_configurable(self):
        """测试阈值可由用户配置
        Test the thresholds can be overridden by the user"""
        with (
            mock.patch.object(
                paddle, "is_compiled_with_cuda", return_value=True
            ),
            mock.patch.object(
                paddle.device.cuda, "get_device_capability", return_value=(9, 0)
            ),
            mock.patch.object(
                muon_module,
                "_load_symmetric_gemm",
                mock.Mock(return_value=fake_gemm_symmetric),
            ),
        ):
            opt = Muon(
                parameters=[self._param()],
                use_symmetric_gemm=True,
                ns_matmul_dtype=paddle.bfloat16,
                symmetric_gemm_min_short_edge=512,
                symmetric_gemm_min_step_flops=1e9,
            )
        self.assertEqual(opt._symmetric_gemm_min_short_edge, 512)
        self.assertEqual(opt._symmetric_gemm_min_step_flops, 1e9)

    def test_rejects_non_positive_thresholds(self):
        """测试非正阈值被拒绝
        Test non-positive thresholds are rejected"""
        for kwargs in (
            {"symmetric_gemm_min_short_edge": 0},
            {"symmetric_gemm_min_short_edge": -1},
            {"symmetric_gemm_min_step_flops": 0},
            {"symmetric_gemm_min_step_flops": -1e9},
        ):
            with (
                mock.patch.object(
                    paddle, "is_compiled_with_cuda", return_value=True
                ),
                mock.patch.object(
                    paddle.device.cuda,
                    "get_device_capability",
                    return_value=(9, 0),
                ),
                self.assertRaises(ValueError) as ctx,
            ):
                Muon(
                    parameters=[self._param()],
                    use_symmetric_gemm=True,
                    ns_matmul_dtype=paddle.bfloat16,
                    **kwargs,
                )
            self.assertIn("must be positive", str(ctx.exception))

    def test_thresholds_unvalidated_when_path_disabled(self):
        """测试未开启对称 GEMM 时不校验阈值
        Test the thresholds are not validated while the path is disabled"""
        opt = Muon(
            parameters=[self._param()],
            symmetric_gemm_min_short_edge=0,
            symmetric_gemm_min_step_flops=0,
        )
        self.assertFalse(opt._use_symmetric_gemm)


class TestSymmetricNewtonSchulzStep(unittest.TestCase):
    """测试对称 Newton-Schulz 迭代步
    Test the symmetric Newton-Schulz step implementations"""

    COEFFS = (3.4445, -4.7750, 2.0315)

    def setUp(self):
        paddle.seed(2026)
        self.loader = mock.patch.object(
            muon_module,
            "_load_symmetric_gemm",
            return_value=fake_gemm_symmetric,
        )
        self.loader.start()
        self.addCleanup(self.loader.stop)

    def test_2d_step_matches_dense_step(self):
        """测试 2D 对称步与稠密步数值一致
        Test the 2D symmetric step matches the dense step"""
        x = paddle.randn([8, 16], dtype='float32') / 4.0
        got = Muon._newton_schulz_step_symmetric(x, *self.COEFFS)
        ref = Muon._newton_schulz_step(x, *self.COEFFS)
        np.testing.assert_allclose(
            got.numpy(), ref.numpy(), rtol=1e-5, atol=1e-6
        )

    def test_3d_step_matches_dense_step(self):
        """测试 3D 对称步与稠密步数值一致
        Test the batched symmetric step matches the dense step"""
        x = paddle.randn([3, 8, 16], dtype='float32') / 4.0
        got = Muon._batched_newton_schulz_step_symmetric(x, *self.COEFFS)
        ref = Muon._batched_newton_schulz_step(x, *self.COEFFS)
        np.testing.assert_allclose(
            got.numpy(), ref.numpy(), rtol=1e-5, atol=1e-6
        )


class TestZeropowerSymmetricDispatch(unittest.TestCase):
    """测试 _zeropower_via_newtonschulz5 的对称路径选择
    Test symmetric-path dispatch inside _zeropower_via_newtonschulz5"""

    def setUp(self):
        paddle.seed(2026)
        self.calls = []

        def counting_gemm(A, B, C=None, alpha=1.0, beta=0.0):
            self.calls.append(tuple(A.shape))
            return fake_gemm_symmetric(A, B, C=C, alpha=alpha, beta=beta)

        patches = [
            mock.patch.object(
                muon_module, "_load_symmetric_gemm", return_value=counting_gemm
            ),
            # fp32 never takes this path in production; allow it here so the
            # dispatch can be exercised on a CPU-sized tensor.
            mock.patch.object(
                muon_module, "_SYRK_SUPPORTED_DTYPES", (paddle.float32,)
            ),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    # Loose enough that the tiny shapes below clear the gate.
    LOOSE = (8, 1)

    def _run(self, shape, use_symmetric_gemm, thresholds=LOOSE):
        x = paddle.randn(shape, dtype='float32')
        return Muon._zeropower_via_newtonschulz5(
            x,
            steps=2,
            ns_matmul_dtype=paddle.float32,
            use_symmetric_gemm=use_symmetric_gemm,
            symmetric_gemm_min_short_edge=thresholds[0],
            symmetric_gemm_min_step_flops=thresholds[1],
        ), x

    def test_2d_takes_symmetric_path(self):
        """测试 2D 输入命中门控时走对称 kernel
        Test a 2D input that clears the gate uses the symmetric kernel"""
        got, x = self._run([8, 16], True)
        self.assertEqual(len(self.calls), 4)
        ref = Muon._zeropower_via_newtonschulz5(
            x, steps=2, ns_matmul_dtype=paddle.float32
        )
        np.testing.assert_allclose(
            got.numpy(), ref.numpy(), rtol=1e-5, atol=1e-6
        )

    def test_3d_takes_symmetric_path(self):
        """测试 3D 输入命中门控时走对称 kernel
        Test a 3D input that clears the gate uses the symmetric kernel"""
        got, x = self._run([2, 8, 16], True)
        self.assertEqual(len(self.calls), 4)
        ref = Muon._zeropower_via_newtonschulz5(
            x, steps=2, ns_matmul_dtype=paddle.float32
        )
        np.testing.assert_allclose(
            got.numpy(), ref.numpy(), rtol=1e-5, atol=1e-6
        )

    def test_disabled_flag_keeps_dense_path(self):
        """测试 use_symmetric_gemm=False 时不调用对称 kernel
        Test the symmetric kernel is untouched when the flag is off"""
        self._run([8, 16], False)
        self.assertEqual(self.calls, [])

    def test_unprofitable_shape_keeps_dense_path(self):
        """测试形状未过门控时回退到 cuBLAS 路径
        Test a shape that fails the gate falls back to the dense path"""
        self._run([8, 16], True, thresholds=SYRK_DEFAULTS)
        self.assertEqual(self.calls, [])

    def test_each_threshold_gates_independently(self):
        """测试两个阈值各自都能拦住对称路径
        Test either threshold on its own can keep the dense path

        Also covers that both values are really threaded through
        ``_zeropower_via_newtonschulz5`` rather than read from elsewhere.
        """
        self._run([8, 16], True, thresholds=(1024, 1))
        self.assertEqual(self.calls, [])
        self._run([8, 16], True, thresholds=(8, 1e30))
        self.assertEqual(self.calls, [])
        self._run([8, 16], True, thresholds=(8, 1))
        self.assertEqual(len(self.calls), 4)


class TestSymmetricGemmRealKernel(unittest.TestCase):
    """测试真实 quack kernel（生产路径，非 mock）
    Test the real quack kernel, i.e. the production path, unmocked

    Every other symmetric-GEMM test replaces ``gemm_symmetric`` with a dense
    stand-in, so only this class covers ``use_compat_guard`` importing, the
    Paddle/DLPack hand-off and the sm90/100/110 kernel itself. It skips where
    that hardware or quack is missing -- set
    ``PADDLE_MUON_REQUIRE_SYMMETRIC_GEMM=1`` on the pipeline that owns this
    coverage to turn the skip into a failure so it cannot go unnoticed.
    """

    REQUIRE_ENV = "PADDLE_MUON_REQUIRE_SYMMETRIC_GEMM"

    def setUp(self):
        blocker = quack_symmetric_gemm_blocker()
        if blocker is not None:
            if os.environ.get(self.REQUIRE_ENV) == "1":
                self.fail(
                    f"{self.REQUIRE_ENV}=1 demands the real quack symmetric "
                    f"GEMM regression, but it cannot run: {blocker}"
                )
            self.skipTest(f"real quack symmetric GEMM unavailable: {blocker}")
        paddle.seed(2026)

    def test_kernel_output_is_exactly_symmetric(self):
        """测试 kernel 的镜像写回给出严格对称结果
        Test the mirror write-back yields an exactly symmetric result"""
        gemm_symmetric = _load_symmetric_gemm()
        shape = [4, 2048, 4096]
        x = (paddle.randn(shape, dtype='float32') / shape[-1] ** 0.5).astype(
            'bfloat16'
        )
        out = gemm_symmetric(x, paddle.transpose(x, perm=[0, 2, 1]))
        ref = paddle.matmul(x, x, transpose_y=True)
        upper = paddle.transpose(out, perm=[0, 2, 1])
        self.assertEqual(
            (out.astype('float32') - upper.astype('float32'))
            .abs()
            .max()
            .item(),
            0.0,
        )
        np.testing.assert_allclose(
            out.astype('float32').numpy(),
            ref.astype('float32').numpy(),
            rtol=5e-3,
            atol=5e-3,
        )

    def test_gate_rejects_shapes_the_kernel_cannot_take(self):
        """测试门控拒绝的非对齐形状确实会让 kernel 报错
        Test the shapes the gate rejects really do break the kernel

        Guards the gate against drift: if quack ever relaxes its alignment
        rules this fails, and the gate can be loosened deliberately.
        """
        gemm_symmetric = _load_symmetric_gemm()
        for shape in ([1024, 4100], [1028, 4096]):
            self.assertFalse(
                _symmetric_gemm_is_profitable(
                    shape, paddle.bfloat16, *SYRK_DEFAULTS
                ),
                msg=str(shape),
            )
            x = paddle.randn(shape, dtype='float32').astype('bfloat16')
            with self.assertRaises(ValueError):
                gemm_symmetric(x, paddle.transpose(x, perm=[1, 0]))

    def test_newton_schulz_matches_cublas(self):
        """测试 5 步 NS 迭代在两条路径上结果一致
        Test a 5-step NS iteration agrees between both paths"""
        shape = [16, 2048, 2048]
        x = (paddle.randn(shape, dtype='float32') / shape[-1] ** 0.5).astype(
            'bfloat16'
        )
        ref = Muon._zeropower_via_newtonschulz5(
            x, steps=5, ns_matmul_dtype=paddle.bfloat16
        )
        got = Muon._zeropower_via_newtonschulz5(
            x,
            steps=5,
            ns_matmul_dtype=paddle.bfloat16,
            use_symmetric_gemm=True,
            symmetric_gemm_min_short_edge=SYRK_DEFAULTS[0],
            symmetric_gemm_min_step_flops=SYRK_DEFAULTS[1],
        )
        np.testing.assert_allclose(
            got.astype('float32').numpy(),
            ref.astype('float32').numpy(),
            rtol=5e-3,
            atol=5e-3,
        )

    def test_optimizer_step_matches_cublas(self):
        """测试整步优化器更新在两条路径上结果一致
        Test a full optimizer step agrees between both paths"""
        updates = []
        for use_symmetric_gemm in (False, True):
            paddle.seed(2026)
            p = paddle.create_parameter(shape=[4096, 8192], dtype='float32')
            p.stop_gradient = False
            opt = Muon(
                learning_rate=0.02,
                parameters=[p],
                muon_exclude_patterns=['embed', 'bias', 'lm_head'],
                use_symmetric_gemm=use_symmetric_gemm,
            )
            before = p.numpy().copy()
            p.grad = paddle.randn(p.shape, dtype='float32') * 0.01
            opt.step()
            self.assertTrue(bool(paddle.isfinite(p).all()))
            updates.append(p.numpy() - before)
        self.assertGreater(np.abs(updates[0]).max(), 0.0)
        np.testing.assert_allclose(updates[1], updates[0], rtol=1e-3, atol=1e-6)


def _fro_np(x, axis=None):
    return np.sqrt(np.sum(np.square(x.astype(np.float64)), axis=axis))


class TestMuonParamInfoHyperball(unittest.TestCase):
    """测试 MuonParamInfo.use_hyperball 字段
    Test the MuonParamInfo.use_hyperball field"""

    def test_use_hyperball_defaults_false(self):
        """默认 use_hyperball 为 False（向后兼容：老代码不受影响）
        use_hyperball defaults to False (backward compatible)"""
        info = MuonParamInfo()
        self.assertFalse(info.use_hyperball)

    def test_use_hyperball_custom_true(self):
        """可显式设置 use_hyperball=True
        use_hyperball can be set to True explicitly"""
        info = MuonParamInfo(use_muon=True, use_hyperball=True)
        self.assertTrue(info.use_muon)
        self.assertTrue(info.use_hyperball)

    def test_all_four_modes_representable(self):
        """(use_muon, use_hyperball) 两个正交布尔能组合出四种模式
        The two orthogonal booleans represent all four modes"""
        modes = {
            (True, True): "MuonH",
            (False, True): "AdamH",
            (True, False): "Muon",
            (False, False): "Adam",
        }
        for um, uh in modes:
            info = MuonParamInfo(use_muon=um, use_hyperball=uh)
            self.assertEqual((info.use_muon, info.use_hyperball), (um, uh))


class TestHyperballApply(unittest.TestCase):
    """测试 Muon._hyperball_apply 静态方法（球面投影收尾）
    Test the Muon._hyperball_apply static method (Frobenius-sphere projection)"""

    def test_2d_preserves_frobenius_norm(self):
        """2D：投影后 ||W||_F 恒等于 R=||w||_F
        2D: the projection keeps ||W||_F equal to R=||w||_F"""
        paddle.seed(2026)
        w = paddle.randn([6, 4], dtype='float32')
        u = paddle.randn([6, 4], dtype='float32')
        out = Muon._hyperball_apply(w, u, 0.1)
        self.assertAlmostEqual(
            _fro_np(out.numpy()), _fro_np(w.numpy()), places=4
        )

    def test_3d_norm_is_per_expert(self):
        """3D [E,H,I]：范数按最后两维逐专家保持
        3D [E,H,I]: the norm is preserved per expert (last two dims)"""
        paddle.seed(2026)
        w = paddle.randn([3, 4, 5], dtype='float32')
        u = paddle.randn([3, 4, 5], dtype='float32')
        out = Muon._hyperball_apply(w, u, 0.1)
        r0 = _fro_np(w.numpy(), axis=(-2, -1))
        r1 = _fro_np(out.numpy(), axis=(-2, -1))
        np.testing.assert_allclose(r1, r0, rtol=1e-4, atol=1e-4)

    def test_formula_matches_numpy_reference(self):
        """与 R*Normalize(w - lr*R*Normalize(u)) 的独立 numpy 参考逐元素一致
        Matches an independent numpy reference of the Eq.(1) update"""
        paddle.seed(7)
        w = paddle.randn([5, 3], dtype='float32')
        u = paddle.randn([5, 3], dtype='float32')
        lr = 0.2
        wn, un = w.numpy().astype(np.float64), u.numpy().astype(np.float64)
        R = _fro_np(wn)
        trial = wn - lr * R * (un / _fro_np(un))
        ref = R * trial / _fro_np(trial)
        out = Muon._hyperball_apply(w, u, lr)
        np.testing.assert_allclose(out.numpy(), ref, rtol=1e-4, atol=1e-5)

    def test_zero_lr_returns_w(self):
        """lr=0 时 R*Normalize(w)==w（w 本就在半径 R 的球面上）
        lr=0 gives R*Normalize(w)==w (w already sits on the sphere)"""
        paddle.seed(1)
        w = paddle.randn([4, 4], dtype='float32')
        out = Muon._hyperball_apply(
            w, paddle.randn([4, 4], dtype='float32'), 0.0
        )
        np.testing.assert_allclose(out.numpy(), w.numpy(), rtol=1e-5, atol=1e-5)

    def test_eps_guards_zero_update(self):
        """update 全零时 eps 防止除零，输出有限且范数仍为 R
        A zero update must not divide by zero; output stays finite at norm R"""
        w = paddle.ones([3, 3], dtype='float32')
        out = Muon._hyperball_apply(
            w, paddle.zeros([3, 3], dtype='float32'), 0.1
        )
        self.assertTrue(bool(paddle.isfinite(out).all()))
        self.assertAlmostEqual(
            _fro_np(out.numpy()), _fro_np(w.numpy()), places=4
        )

    def test_formula_matches_reference_2d_and_3d(self):
        """2D 整张、3D 逐专家都逐元素对齐独立 numpy 参考
        Both 2D (whole matrix) and 3D (per expert) match a numpy reference"""
        paddle.seed(11)
        lr = 0.3
        for shape in ([5, 3], [4, 3, 6], [2, 8, 8]):
            w = paddle.randn(shape, dtype='float32')
            u = paddle.randn(shape, dtype='float32')
            wn = w.numpy().astype(np.float64)
            un = u.numpy().astype(np.float64)
            ax = (-2, -1)
            R = _fro_np(wn, axis=ax)
            if wn.ndim == 3:
                R = R[:, None, None]
                nu = _fro_np(un, axis=ax)[:, None, None]
            else:
                nu = _fro_np(un, axis=ax)
            trial = wn - lr * R * (un / nu)
            nt = _fro_np(trial, axis=ax)
            if wn.ndim == 3:
                nt = nt[:, None, None]
            ref = R * trial / nt
            out = Muon._hyperball_apply(w, u, lr)
            np.testing.assert_allclose(
                out.numpy(), ref, rtol=1e-4, atol=1e-5, err_msg=str(shape)
            )

    def test_3d_experts_are_independent(self):
        """3D：逐专家独立——放大某个专家不影响其它专家的输出，
        且每个专家仍各自保持自己的 R。
        3D per-expert independence: scaling one expert leaves the others'
        output untouched, and every expert keeps its own R."""
        paddle.seed(12)
        w = paddle.randn([3, 4, 5], dtype='float32')
        u = paddle.randn([3, 4, 5], dtype='float32')
        out = Muon._hyperball_apply(w, u, 0.2)
        # scale ONLY expert 0 by 100x, everything else identical
        w2 = w.clone()
        w2[0] = w2[0] * 100.0
        out2 = Muon._hyperball_apply(w2, u, 0.2)
        # experts 1 and 2 are byte-identical (no cross-expert leakage)
        np.testing.assert_array_equal(out2.numpy()[1:], out.numpy()[1:])
        # each expert preserves its own (possibly new) radius
        r_in = _fro_np(w2.numpy(), axis=(-2, -1))
        r_out = _fro_np(out2.numpy(), axis=(-2, -1))
        np.testing.assert_allclose(r_out, r_in, rtol=1e-4, atol=1e-4)

    def test_2d_non_square(self):
        """2D 非方阵（高/宽矩阵）都保持整张 Frobenius 范数
        Non-square 2D matrices (tall/wide) preserve the whole-matrix norm"""
        paddle.seed(13)
        for shape in ([8, 3], [3, 8], [1, 16], [16, 1]):
            w = paddle.randn(shape, dtype='float32')
            u = paddle.randn(shape, dtype='float32')
            out = Muon._hyperball_apply(w, u, 0.15)
            self.assertEqual(list(out.shape), shape)
            self.assertAlmostEqual(
                _fro_np(out.numpy()), _fro_np(w.numpy()), places=4, msg=str(shape)
            )


class TestFrobeniusNormEquivalence(unittest.TestCase):
    """_fro 的手写实现 vs paddle 范数 API：精度等价 + 速度不劣化。

    Hyperball 的 _fro 用 ``sqrt(sum(x*x, axis=[-2,-1]))`` 手写。本类证明它与
    ``paddle.linalg.norm`` / ``paddle.norm`` / ``Tensor.norm`` 的 Frobenius 范数
    数值等价（2D 整张、3D 逐专家），故实现可安全互换；并对速度做一个宽松的健全性保护。
    """

    def _manual(self, x):
        return paddle.sqrt(paddle.sum(x * x, axis=[-2, -1], keepdim=True))

    def test_precision_matches_apis_fp32(self):
        """fp32 下手写 == linalg.norm == paddle.norm == tensor.norm（2D/3D）
        fp32: manual matches all three fro-norm APIs, shape included"""
        paddle.seed(21)
        for shape in ([6, 4], [3, 8], [3, 4, 5], [2, 16, 16]):
            x = paddle.randn(shape, dtype='float32')
            ref = self._manual(x).numpy()
            apis = {
                "linalg.norm": paddle.linalg.norm(
                    x, p='fro', axis=[-2, -1], keepdim=True
                ),
                "paddle.norm": paddle.norm(
                    x, p='fro', axis=[-2, -1], keepdim=True
                ),
                "tensor.norm": x.norm(p='fro', axis=[-2, -1], keepdim=True),
            }
            for name, got in apis.items():
                self.assertEqual(
                    list(got.shape), list(ref.shape), msg=f"{shape} {name} shape"
                )
                np.testing.assert_allclose(
                    got.numpy(), ref, rtol=1e-5, atol=1e-5,
                    err_msg=f"{shape} {name}",
                )

    def test_precision_exact_fp64(self):
        """fp64 下手写与 linalg.norm 高精度一致（排除 fp32 舍入干扰）
        fp64: manual and linalg.norm agree to ~1e-12"""
        paddle.seed(22)
        for shape in ([6, 4], [3, 4, 5]):
            x = paddle.randn(shape, dtype='float64')
            ref = self._manual(x).numpy()
            got = paddle.linalg.norm(
                x, p='fro', axis=[-2, -1], keepdim=True
            ).numpy()
            np.testing.assert_allclose(
                got, ref, rtol=1e-12, atol=1e-12, err_msg=str(shape)
            )

    @unittest.skipUnless(
        paddle.is_compiled_with_cuda(), "speed sanity check is GPU-only"
    )
    def test_speed_api_not_pathologically_slower(self):
        """宽松健全性保护（非严格性能门）：fro 范数 API 不应比手写慢 5 倍以上。
        实测 linalg.norm/tensor.norm 反而更快（融合、无全尺寸 x*x 中间量）。
        Loose guard against a pathological (e.g. SVD-based) regression, not a
        tight perf SLA; in practice the API is ~2x faster than the manual form.
        """
        import time

        x = paddle.randn([4096, 4096], dtype='float32')

        def bench(fn, n=300):
            for _ in range(20):
                fn()
            paddle.device.synchronize()
            t = time.time()
            for _ in range(n):
                fn()
            paddle.device.synchronize()
            return (time.time() - t) / n

        manual_t = bench(lambda: self._manual(x))
        api_t = bench(
            lambda: paddle.linalg.norm(x, p='fro', axis=[-2, -1], keepdim=True)
        )
        self.assertLess(
            api_t,
            manual_t * 5.0,
            msg=f"manual={manual_t * 1e6:.1f}us api={api_t * 1e6:.1f}us",
        )


def _fro(t):
    return float((t.astype('float32') ** 2).sum().sqrt())


class TestMuonHyperballStep(unittest.TestCase):
    """测试 MuonH（Muon 方向 + 球面投影）整步更新
    Test a full MuonH (Muon direction + sphere projection) step"""

    def _run_step(self, weight_decay, lr_ratio=None):
        paddle.seed(2026)
        p = paddle.create_parameter(shape=[6, 4], dtype='float32')
        p.set_value(np.random.RandomState(0).randn(6, 4).astype('float32'))
        opt = Muon(
            parameters=[p],
            learning_rate=0.05,
            weight_decay=weight_decay,
            ns_steps=3,
            ns_matmul_dtype=paddle.float32,
            muon_param_info_map={
                p.name: MuonParamInfo(use_muon=True, use_hyperball=True)
            },
            lr_ratio=lr_ratio,
        )
        r0 = _fro(p)
        p.grad = paddle.to_tensor(
            np.random.RandomState(1).randn(6, 4).astype('float32')
        )
        opt.step()
        return p, r0

    def test_muonh_preserves_norm(self):
        """MuonH 一步后 ||W||_F 不变（球面约束）
        After one MuonH step ||W||_F is unchanged (sphere constraint)"""
        p, r0 = self._run_step(weight_decay=0.1)
        self.assertAlmostEqual(_fro(p), r0, places=3)

    def test_muonh_drops_weight_decay(self):
        """hyperball 丢弃 WD：不同 weight_decay 得到相同结果
        Hyperball drops weight decay: different wd -> identical result"""
        p_a, _ = self._run_step(weight_decay=0.0)
        p_b, _ = self._run_step(weight_decay=0.5)
        np.testing.assert_allclose(
            p_a.numpy(), p_b.numpy(), rtol=1e-6, atol=1e-6
        )

    def test_muonh_lr_ratio_zero_freezes(self):
        """lr_ratio=0 冻结参数（effective_lr=lr*ratio 生效）
        lr_ratio=0 freezes the param (effective_lr=lr*ratio applies)"""
        p0, _ = self._run_step(weight_decay=0.1, lr_ratio=lambda _: 0.0)
        expected = np.random.RandomState(0).randn(6, 4).astype('float32')
        np.testing.assert_allclose(p0.numpy(), expected, rtol=1e-5, atol=1e-5)

    @unittest.skipUnless(
        paddle.is_compiled_with_cuda(), "fp16 master-weight path needs CUDA"
    )
    def test_muonh_master_weight_path(self):
        """multi_precision + fp16：走 master weight 分支，fp32 master 上 ||W||_F 守恒
        fp16 + multi_precision exercises the master-weight branch; the norm is
        preserved on the fp32 master across steps"""
        paddle.seed(2026)
        p = paddle.create_parameter(shape=[6, 4], dtype='float16')
        p.set_value(
            (np.random.RandomState(0).randn(6, 4) * 0.1).astype('float16')
        )
        opt = Muon(
            parameters=[p],
            learning_rate=0.05,
            weight_decay=0.1,
            ns_steps=3,
            ns_matmul_dtype=paddle.float32,
            multi_precision=True,
            muon_param_info_map={
                p.name: MuonParamInfo(use_muon=True, use_hyperball=True)
            },
        )
        p.grad = paddle.to_tensor(
            (np.random.RandomState(1).randn(6, 4) * 0.1).astype('float16')
        )
        opt.step()
        self.assertIn(p.name, opt._master_weights)
        r1 = _fro(opt._master_weights[p.name])
        p.grad = paddle.to_tensor(
            (np.random.RandomState(2).randn(6, 4) * 0.1).astype('float16')
        )
        opt.step()
        self.assertAlmostEqual(_fro(opt._master_weights[p.name]), r1, places=3)
        self.assertTrue(bool(paddle.isfinite(p).all()))

    def test_muonh_3d_per_expert_preserved(self):
        """3D 专家权重 [E,H,I]：MuonH 一步后每个专家各自的 ||W||_F 守恒
        3D grouped-expert weight: after one MuonH step each expert keeps its
        own ||W||_F (batched Newton-Schulz + per-expert projection)"""
        paddle.seed(2026)
        p = paddle.create_parameter(shape=[3, 4, 5], dtype='float32')
        p.set_value(np.random.RandomState(0).randn(3, 4, 5).astype('float32'))
        r0 = _fro_np(p.numpy(), axis=(-2, -1))
        opt = Muon(
            parameters=[p],
            learning_rate=0.05,
            weight_decay=0.1,
            ns_steps=3,
            ns_matmul_dtype=paddle.float32,
            muon_param_info_map={
                p.name: MuonParamInfo(use_muon=True, use_hyperball=True)
            },
        )
        p.grad = paddle.to_tensor(
            np.random.RandomState(1).randn(3, 4, 5).astype('float32')
        )
        opt.step()
        np.testing.assert_allclose(
            _fro_np(p.numpy(), axis=(-2, -1)), r0, rtol=1e-3, atol=1e-3
        )


class TestAdamHStep(unittest.TestCase):
    """测试 AdamH（Adam 方向 + 球面投影）整步更新
    Test a full AdamH (Adam direction + sphere projection) step"""

    def _build(self, weight_decay):
        paddle.seed(2026)
        p = paddle.create_parameter(shape=[5, 5], dtype='float32')
        p.set_value(np.random.RandomState(0).randn(5, 5).astype('float32'))
        opt = Muon(
            parameters=[p],
            learning_rate=0.05,
            weight_decay=weight_decay,
            adam_beta1=0.9,
            adam_beta2=0.95,
            ns_matmul_dtype=paddle.float32,
            muon_param_info_map={
                p.name: MuonParamInfo(use_muon=False, use_hyperball=True)
            },
        )
        p.grad = paddle.to_tensor(
            np.random.RandomState(1).randn(5, 5).astype('float32')
        )
        return p, opt

    def test_adamh_preserves_norm(self):
        """AdamH 一步后 ||W||_F 不变
        After one AdamH step ||W||_F is unchanged"""
        p, opt = self._build(weight_decay=0.1)
        r0 = _fro(p)
        opt.step()
        self.assertAlmostEqual(_fro(p), r0, places=3)

    def test_adamh_creates_adam_accumulators(self):
        """AdamH 建 moment1+moment2+beta_pow（与普通 AdamW 相同状态）
        AdamH creates moment1+moment2+beta_pow (same state as plain AdamW)"""
        p, opt = self._build(weight_decay=0.1)
        opt.step()
        acc = opt._accumulators
        self.assertIn(opt._moment_acc_str, acc)
        self.assertIn(opt._moment2_acc_str, acc)
        self.assertIn(p.name, acc[opt._moment2_acc_str])

    def test_adamh_advances_beta_pow(self):
        """AdamH 推进 beta1_pow（步数状态一致）：一步后从初始 beta1 前进
        AdamH advances beta1_pow: after one step it moves off the initial beta1"""
        p, opt = self._build(weight_decay=0.1)
        opt.step()  # accumulators are created during the first step
        b1 = float(opt._get_accumulator(opt._beta1_pow_acc_str, p))
        # initial fill is adam_beta1=0.9; after one step it is advanced (*=beta1).
        self.assertFalse(abs(b1 - 0.9) < 1e-6)

    def test_adamh_drops_weight_decay(self):
        """AdamH 丢弃 WD：不同 weight_decay 结果一致
        AdamH drops weight decay: different wd -> identical result"""
        pa, oa = self._build(weight_decay=0.0)
        pb, ob = self._build(weight_decay=0.7)
        oa.step()
        ob.step()
        np.testing.assert_allclose(pa.numpy(), pb.numpy(), rtol=1e-6, atol=1e-6)

    @unittest.skipUnless(
        paddle.is_compiled_with_cuda(), "fp16 master-weight path needs CUDA"
    )
    def test_adamh_master_weight_path(self):
        """multi_precision + fp16：AdamH 走 master weight 分支，master 上范数守恒
        fp16 + multi_precision exercises AdamH's master-weight branch; the norm
        is preserved on the fp32 master across steps"""
        paddle.seed(2026)
        p = paddle.create_parameter(shape=[5, 5], dtype='float16')
        p.set_value(
            (np.random.RandomState(0).randn(5, 5) * 0.1).astype('float16')
        )
        opt = Muon(
            parameters=[p],
            learning_rate=0.05,
            weight_decay=0.1,
            adam_beta1=0.9,
            adam_beta2=0.95,
            ns_matmul_dtype=paddle.float32,
            multi_precision=True,
            muon_param_info_map={
                p.name: MuonParamInfo(use_muon=False, use_hyperball=True)
            },
        )
        p.grad = paddle.to_tensor(
            (np.random.RandomState(1).randn(5, 5) * 0.1).astype('float16')
        )
        opt.step()
        self.assertIn(p.name, opt._master_weights)
        r1 = _fro(opt._master_weights[p.name])
        p.grad = paddle.to_tensor(
            (np.random.RandomState(2).randn(5, 5) * 0.1).astype('float16')
        )
        opt.step()
        self.assertAlmostEqual(_fro(opt._master_weights[p.name]), r1, places=3)
        self.assertTrue(bool(paddle.isfinite(p).all()))

    def test_adamh_3d_per_expert_preserved(self):
        """3D 专家权重 [E,H,I]：AdamH 一步后每个专家各自的 ||W||_F 守恒
        3D grouped-expert weight: after one AdamH step each expert keeps its
        own ||W||_F"""
        paddle.seed(2026)
        p = paddle.create_parameter(shape=[3, 4, 5], dtype='float32')
        p.set_value(np.random.RandomState(0).randn(3, 4, 5).astype('float32'))
        r0 = _fro_np(p.numpy(), axis=(-2, -1))
        opt = Muon(
            parameters=[p],
            learning_rate=0.05,
            weight_decay=0.1,
            adam_beta1=0.9,
            adam_beta2=0.95,
            ns_matmul_dtype=paddle.float32,
            muon_param_info_map={
                p.name: MuonParamInfo(use_muon=False, use_hyperball=True)
            },
        )
        p.grad = paddle.to_tensor(
            np.random.RandomState(1).randn(3, 4, 5).astype('float32')
        )
        opt.step()
        np.testing.assert_allclose(
            _fro_np(p.numpy(), axis=(-2, -1)), r0, rtol=1e-3, atol=1e-3
        )


class TestApplyOptimizeRouting(unittest.TestCase):
    """测试 _apply_optimize 按 (use_muon, use_hyperball) 路由到四条路径
    Test _apply_optimize routes the four modes in a single step"""

    def test_four_modes_in_one_step(self):
        """一次 step 中 MuonH/AdamH/Muon/Adam 同批更新，
        两个 hyperball 参数各自保持 ||W||_F，四者都被更新。
        MuonH/AdamH/Muon/Adam updated together; the two hyperball params keep
        their norm; all four move."""
        paddle.seed(2026)
        rs = np.random.RandomState(0)
        params, info, before = {}, {}, {}
        specs = {
            "muonh": (True, True),
            "adamh": (False, True),
            "muon": (True, False),
            "adam": (False, False),
        }
        plist = []
        for key, (um, uh) in specs.items():
            p = paddle.create_parameter(shape=[4, 4], dtype='float32')
            p.set_value(rs.randn(4, 4).astype('float32'))
            p.grad = paddle.to_tensor(rs.randn(4, 4).astype('float32') * 0.1)
            params[key] = p
            info[p.name] = MuonParamInfo(use_muon=um, use_hyperball=uh)
            before[key] = p.numpy().copy()
            plist.append(p)
        opt = Muon(
            parameters=plist,
            learning_rate=0.05,
            weight_decay=0.1,
            ns_steps=3,
            ns_matmul_dtype=paddle.float32,
            muon_param_info_map=info,
        )
        opt.step()
        for key in specs:
            self.assertTrue(bool(paddle.isfinite(params[key]).all()))
            self.assertFalse(
                np.array_equal(params[key].numpy(), before[key]),
                msg=f"{key} did not move",
            )
        for key in ("muonh", "adamh"):
            self.assertAlmostEqual(
                _fro(params[key]), _fro_np(before[key]), places=3
            )


class TestMuonEpsilon(unittest.TestCase):
    """测试 per-optimizer muon_epsilon（NS 方向 eps 与 adam_epsilon 解耦）
    Test the per-optimizer muon_epsilon (NS eps decoupled from adam_epsilon)"""

    def _param(self):
        p = paddle.create_parameter(shape=[4, 4], dtype='float32')
        p.stop_gradient = False
        return p

    def test_default_falls_back_to_adam_epsilon(self):
        """不设 muon_epsilon 时回落到 adam_epsilon（向后兼容的共享行为）
        Unset muon_epsilon falls back to adam_epsilon (shared, backward compat)"""
        opt = Muon(
            parameters=[self._param()],
            adam_epsilon=1e-9,
            ns_matmul_dtype=paddle.float32,
        )
        self.assertEqual(opt._default_dict["muon_epsilon"], 1e-9)
        self.assertEqual(opt._default_dict["epsilon"], 1e-9)

    def test_explicit_value_is_independent(self):
        """显式 muon_epsilon 与 adam_epsilon 相互独立
        Explicit muon_epsilon is independent of adam_epsilon"""
        opt = Muon(
            parameters=[self._param()],
            adam_epsilon=1e-15,
            muon_epsilon=1e-7,
            ns_matmul_dtype=paddle.float32,
        )
        self.assertEqual(opt._default_dict["muon_epsilon"], 1e-7)
        self.assertEqual(opt._default_dict["epsilon"], 1e-15)

    def test_muon_step_runs_with_custom_epsilon(self):
        """带自定义 muon_epsilon 的 Muon 方向整步更新可运行且结果有限
        A Muon-direction step with a custom muon_epsilon runs and stays finite"""
        paddle.seed(2026)
        p = self._param()
        p.set_value(np.random.RandomState(0).randn(4, 4).astype('float32'))
        opt = Muon(
            parameters=[p],
            learning_rate=0.02,
            adam_epsilon=1e-15,
            muon_epsilon=1e-7,
            ns_steps=3,
            ns_matmul_dtype=paddle.float32,
            muon_param_info_map={p.name: MuonParamInfo(use_muon=True)},
        )
        p.grad = paddle.to_tensor(
            np.random.RandomState(1).randn(4, 4).astype('float32')
        )
        opt.step()
        self.assertTrue(bool(paddle.isfinite(p).all()))


if __name__ == '__main__':
    unittest.main()
