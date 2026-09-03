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

# [AUTO-GENERATED] test for paddle/tensor/creation.py
# Target file: python/paddle/tensor/creation.py
# Coverage: 77.2% (764/990) - Uncovered lines subset targeted
# 本文件为 tensor/creation.py 的单元测试 / Unit tests for tensor/creation.py
#
# 测试目标：
# - _complex_to_real_dtype / _real_to_complex_dtype 类型转换
# - empty / empty_like 创建未初始化张量
# - diagflat / diag_embed 对角线操作
# - tril_ / triu_ 原地三角操作
# - linspace / logspace / arange 数值序列
# - assign 赋值操作
# - to_tensor 多种输入类型
# - full / fill_constant / zeros / ones 基础创建

import unittest

import numpy as np

import paddle


class TestComplexRealDtypeConversion(unittest.TestCase):
    """复数-实数类型转换测试 / Complex-real dtype conversion tests"""

    def setUp(self):
        """测试前准备 / Setup before tests"""
        pass

    def tearDown(self):
        """测试后清理 / Teardown after tests"""
        pass

    def test_complex_to_real_complex64(self):
        """测试 COMPLEX64 转 FP32 / Test COMPLEX64 to FP32"""
        from paddle.tensor.creation import _complex_to_real_dtype

        result = _complex_to_real_dtype(paddle.core.VarDesc.VarType.COMPLEX64)
        self.assertEqual(result, paddle.core.VarDesc.VarType.FP32)

    def test_complex_to_real_complex128(self):
        """测试 COMPLEX128 转 FP64 / Test COMPLEX128 to FP64"""
        from paddle.tensor.creation import _complex_to_real_dtype

        result = _complex_to_real_dtype(paddle.core.VarDesc.VarType.COMPLEX128)
        self.assertEqual(result, paddle.core.VarDesc.VarType.FP64)

    def test_complex_to_real_passthrough(self):
        """测试非复数类型直接透传 / Test non-complex dtype pass-through"""
        from paddle.tensor.creation import _complex_to_real_dtype

        result = _complex_to_real_dtype(paddle.core.VarDesc.VarType.FP32)
        self.assertEqual(result, paddle.core.VarDesc.VarType.FP32)

        result = _complex_to_real_dtype(paddle.core.VarDesc.VarType.INT32)
        self.assertEqual(result, paddle.core.VarDesc.VarType.INT32)

    def test_real_to_complex_fp32(self):
        """测试 FP32 转 COMPLEX64 / Test FP32 to COMPLEX64"""
        from paddle.tensor.creation import _real_to_complex_dtype

        result = _real_to_complex_dtype(paddle.core.VarDesc.VarType.FP32)
        self.assertEqual(result, paddle.core.VarDesc.VarType.COMPLEX64)

    def test_real_to_complex_fp64(self):
        """测试 FP64 转 COMPLEX128 / Test FP64 to COMPLEX128"""
        from paddle.tensor.creation import _real_to_complex_dtype

        result = _real_to_complex_dtype(paddle.core.VarDesc.VarType.FP64)
        self.assertEqual(result, paddle.core.VarDesc.VarType.COMPLEX128)

    def test_real_to_complex_passthrough(self):
        """测试非浮点类型直接透传 / Test non-float dtype pass-through"""
        from paddle.tensor.creation import _real_to_complex_dtype

        result = _real_to_complex_dtype(paddle.core.VarDesc.VarType.INT32)
        self.assertEqual(result, paddle.core.VarDesc.VarType.INT32)


class TestEmptyCreation(unittest.TestCase):
    """empty / empty_like 创建测试 / empty/empty_like creation tests"""

    def setUp(self):
        """测试前准备 / Setup before tests"""
        paddle.disable_static()

    def tearDown(self):
        """测试后清理 / Teardown after tests"""
        pass

    def test_empty_basic(self):
        """测试创建空张量基本功能 / Test basic empty tensor creation"""
        t = paddle.empty(shape=[3, 4], dtype='float32')
        self.assertEqual(t.shape, [3, 4])
        self.assertEqual(t.dtype, paddle.float32)

    def test_empty_different_dtypes(self):
        """测试不同数据类型的空张量 / Test empty with different dtypes"""
        for dtype in ['float32', 'float64', 'int32', 'int64', 'bool']:
            t = paddle.empty(shape=[2, 3], dtype=dtype)
            self.assertEqual(t.shape, [2, 3])

    def test_empty_zero_shape(self):
        """测试零维空张量 / Test zero-dim empty tensor"""
        t = paddle.empty(shape=[], dtype='float32')
        self.assertEqual(t.shape, [])

    def test_empty_single_element(self):
        """测试单元素空张量 / Test single-element empty tensor"""
        t = paddle.empty(shape=[1], dtype='float32')
        self.assertEqual(t.shape, [1])

    def test_empty_1d(self):
        """测试一维空张量 / Test 1D empty tensor"""
        t = paddle.empty(shape=[5], dtype='float32')
        self.assertEqual(t.shape, [5])

    def test_empty_like_basic(self):
        """测试 empty_like 基本功能 / Test basic empty_like"""
        x = paddle.zeros(shape=[3, 4], dtype='float32')
        t = paddle.empty_like(x)
        self.assertEqual(t.shape, [3, 4])
        self.assertEqual(t.dtype, paddle.float32)

    def test_empty_like_different_dtype(self):
        """测试 empty_like 不同数据类型 / Test empty_like with different dtype"""
        x = paddle.zeros(shape=[2, 3], dtype='float32')
        t = paddle.empty_like(x, dtype='float64')
        self.assertEqual(t.shape, [2, 3])
        self.assertEqual(t.dtype, paddle.float64)

    def test_empty_like_input_alias(self):
        """测试 empty_like 的 input 别名参数 / Test empty_like input alias parameter"""
        x = paddle.zeros(shape=[2, 3], dtype='float32')
        t = paddle.empty_like(input=x)
        self.assertEqual(t.shape, [2, 3])


class TestDiagflatAndDiagEmbed(unittest.TestCase):
    """diagflat / diag_embed 测试 / diagflat/diagembed tests"""

    def setUp(self):
        """测试前准备 / Setup before tests"""
        paddle.disable_static()

    def tearDown(self):
        """测试后清理 / Teardown after tests"""
        pass

    def test_diagflat_1d(self):
        """测试一维输入 diagflat / Test 1D input diagflat"""
        x = paddle.to_tensor([1, 2, 3])
        y = paddle.diagflat(x)
        expected = np.diag([1, 2, 3])
        np.testing.assert_array_equal(y.numpy(), expected)

    def test_diagflat_2d(self):
        """测试二维输入 diagflat / Test 2D input diagflat"""
        x = paddle.to_tensor([[1, 2], [3, 4]])
        y = paddle.diagflat(x)
        expected = np.diagflat([[1, 2], [3, 4]])
        np.testing.assert_array_equal(y.numpy(), expected)

    def test_diagflat_with_offset_positive(self):
        """测试正偏移 diagflat / Test diagflat with positive offset"""
        x = paddle.to_tensor([1, 2, 3])
        y = paddle.diagflat(x, offset=1)
        expected = np.diagflat([1, 2, 3], k=1)
        np.testing.assert_array_equal(y.numpy(), expected)

    def test_diagflat_with_offset_negative(self):
        """测试负偏移 diagflat / Test diagflat with negative offset"""
        x = paddle.to_tensor([1, 2, 3])
        y = paddle.diagflat(x, offset=-1)
        expected = np.diagflat([1, 2, 3], k=-1)
        np.testing.assert_array_equal(y.numpy(), expected)

    def test_diagflat_input_alias(self):
        """测试 diagflat 的 input 别名参数 / Test diagflat input alias"""
        x = paddle.to_tensor([1, 2, 3])
        y = paddle.diagflat(input=x)
        expected = np.diag([1, 2, 3])
        np.testing.assert_array_equal(y.numpy(), expected)

    def test_diag_embed_1d(self):
        """测试一维输入 diag_embed / Test 1D input diag_embed"""
        x = paddle.to_tensor([1, 2, 3], dtype='float32')
        result = paddle.diag_embed(x)
        self.assertEqual(result.shape, [3, 3])
        # 对角线元素应为 [1, 2, 3]
        for i in range(3):
            self.assertEqual(result[i, i].item(), i + 1)

    def test_diag_embed_with_offset(self):
        """测试带偏移的 diag_embed / Test diag_embed with offset"""
        x = paddle.to_tensor([1, 2], dtype='float32')
        result = paddle.diag_embed(x, offset=1)
        self.assertEqual(result.shape, [3, 3])
        # offset=1, 所以对角线在上方
        self.assertEqual(result[0, 1].item(), 1)
        self.assertEqual(result[1, 2].item(), 2)

    def test_diag_embed_2d(self):
        """测试二维输入 diag_embed / Test 2D input diag_embed"""
        x = paddle.to_tensor([[1, 2], [3, 4]], dtype='float32')
        result = paddle.diag_embed(x)
        self.assertEqual(result.shape, [2, 2, 2])

    def test_diag_embed_list_input(self):
        """测试列表输入 diag_embed / Test list input diag_embed"""
        x = [1, 2, 3]
        result = paddle.diag_embed(x)
        self.assertEqual(result.shape, [3, 3])


class TestTrilTriuInplace(unittest.TestCase):
    """tril_ / triu_ 原地三角操作测试 / tril_/triu_ inplace tests"""

    def setUp(self):
        """测试前准备 / Setup before tests"""
        paddle.disable_static()

    def tearDown(self):
        """测试后清理 / Teardown after tests"""
        pass

    def test_tril_inplace_basic(self):
        """测试原地下三角 / Test inplace tril"""
        x = paddle.to_tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype='float32')
        result = paddle.tensor.creation.tril_(x)
        expected = np.tril([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        np.testing.assert_array_equal(result.numpy(), expected)
        # 验证原地操作 / Verify inplace
        np.testing.assert_array_equal(x.numpy(), expected)

    def test_triu_inplace_basic(self):
        """测试原地上三角 / Test inplace triu"""
        x = paddle.to_tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype='float32')
        result = paddle.tensor.creation.triu_(x)
        expected = np.triu([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        np.testing.assert_array_equal(result.numpy(), expected)
        np.testing.assert_array_equal(x.numpy(), expected)

    def test_tril_inplace_with_diagonal(self):
        """测试带偏移的原地下三角 / Test inplace tril with diagonal"""
        x = paddle.to_tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype='float32')
        result = paddle.tensor.creation.tril_(x, diagonal=1)
        expected = np.tril([[1, 2, 3], [4, 5, 6], [7, 8, 9]], k=1)
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_triu_inplace_with_diagonal(self):
        """测试带偏移的原地上三角 / Test inplace triu with diagonal"""
        x = paddle.to_tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype='float32')
        result = paddle.tensor.creation.triu_(x, diagonal=-1)
        expected = np.triu([[1, 2, 3], [4, 5, 6], [7, 8, 9]], k=-1)
        np.testing.assert_array_equal(result.numpy(), expected)


class TestLinspaceAndLogspace(unittest.TestCase):
    """linspace / logspace 测试 / linspace/logspace tests"""

    def setUp(self):
        """测试前准备 / Setup before tests"""
        paddle.disable_static()

    def tearDown(self):
        """测试后清理 / Teardown after tests"""
        pass

    def test_linspace_basic(self):
        """测试基本 linspace / Test basic linspace"""
        result = paddle.linspace(start=0, stop=10, num=5)
        expected = np.linspace(0, 10, 5)
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

    def test_linspace_float_dtype(self):
        """测试 float32 linspace / Test float32 linspace"""
        result = paddle.linspace(start=0, stop=1, num=5, dtype='float32')
        self.assertEqual(result.dtype, paddle.float32)
        expected = np.linspace(0, 1, 5, dtype='float32')
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

    def test_linspace_negative_range(self):
        """测试负范围 linspace / Test negative range linspace"""
        result = paddle.linspace(start=-5, stop=5, num=11)
        expected = np.linspace(-5, 5, 11)
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

    def test_linspace_single_point(self):
        """测试单点 linspace / Test single point linspace"""
        result = paddle.linspace(start=0, stop=10, num=1)
        self.assertEqual(result.shape, [1])

    def test_linspace_float64(self):
        """测试 float64 linspace / Test float64 linspace"""
        result = paddle.linspace(start=0, stop=1, num=5, dtype='float64')
        self.assertEqual(result.dtype, paddle.float64)

    def test_logspace_basic(self):
        """测试基本 logspace / Test basic logspace"""
        result = paddle.logspace(start=0, stop=2, num=5)
        expected = np.logspace(0, 2, 5)
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

    def test_logspace_base_2(self):
        """测试 base=2 logspace / Test base=2 logspace"""
        result = paddle.logspace(start=0, stop=3, num=4, base=2)
        expected = np.logspace(0, 3, 4, base=2)
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

    def test_logspace_float64(self):
        """测试 float64 logspace / Test float64 logspace"""
        result = paddle.logspace(start=0, stop=2, num=5, dtype='float64')
        self.assertEqual(result.dtype, paddle.float64)

    def test_logspace_int_dtype(self):
        """测试 int64 logspace / Test int64 logspace"""
        result = paddle.logspace(start=0, stop=2, num=5, dtype='int64')
        self.assertEqual(result.dtype, paddle.int64)


class TestArange(unittest.TestCase):
    """arange 测试 / arange tests"""

    def setUp(self):
        """测试前准备 / Setup before tests"""
        paddle.disable_static()

    def tearDown(self):
        """测试后清理 / Teardown after tests"""
        pass

    def test_arange_basic(self):
        """测试基本 arange / Test basic arange"""
        result = paddle.arange(5)
        expected = np.arange(5)
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_arange_with_start_stop(self):
        """测试指定 start/stop arange / Test arange with start/stop"""
        result = paddle.arange(start=1, end=5)
        expected = np.arange(1, 5)
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_arange_with_step(self):
        """测试指定 step arange / Test arange with step"""
        result = paddle.arange(start=0, end=10, step=2)
        expected = np.arange(0, 10, 2)
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_arange_float(self):
        """测试浮点数 arange / Test float arange"""
        result = paddle.arange(start=0.0, end=1.0, step=0.2)
        expected = np.arange(0.0, 1.0, 0.2)
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

    def test_arange_negative_step(self):
        """测试负步长 arange / Test negative step arange"""
        result = paddle.arange(start=5, end=0, step=-1)
        expected = np.arange(5, 0, -1)
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_arange_int64_dtype(self):
        """测试 int64 arange / Test int64 arange"""
        result = paddle.arange(start=0, end=5, dtype='int64')
        self.assertEqual(result.dtype, paddle.int64)

    def test_arange_float32_dtype(self):
        """测试 float32 arange / Test float32 arange"""
        result = paddle.arange(start=0, end=5, dtype='float32')
        self.assertEqual(result.dtype, paddle.float32)

    def test_arange_device_alias(self):
        """测试 device 参数别名 / Test device parameter alias"""
        result = paddle.arange(5, device='cpu')
        self.assertEqual(result.shape, [5])

    def test_arange_requires_grad(self):
        """测试 requires_grad 参数 / Test requires_grad parameter"""
        result = paddle.arange(
            start=1, end=5, dtype='float32', requires_grad=True
        )
        self.assertFalse(result.stop_gradient)


class TestAssign(unittest.TestCase):
    """assign 测试 / assign tests"""

    def setUp(self):
        """测试前准备 / Setup before tests"""
        paddle.disable_static()

    def tearDown(self):
        """测试后清理 / Teardown after tests"""
        pass

    def test_assign_list(self):
        """测试列表赋值 / Test list assign"""
        result = paddle.assign([1, 2, 3])
        np.testing.assert_array_equal(result.numpy(), np.array([1, 2, 3]))

    def test_assign_tuple(self):
        """测试元组赋值 / Test tuple assign"""
        result = paddle.assign((4.0, 5.0, 6.0))
        np.testing.assert_array_equal(result.numpy(), np.array([4.0, 5.0, 6.0]))

    def test_assign_ndarray(self):
        """测试 numpy 数组赋值 / Test numpy array assign"""
        data = np.array([[1, 2], [3, 4]])
        result = paddle.assign(data)
        np.testing.assert_array_equal(result.numpy(), data)

    def test_assign_scalar(self):
        """测试标量赋值 / Test scalar assign"""
        result = paddle.assign(42)
        self.assertEqual(result.item(), 42)

    def test_assign_tensor(self):
        """测试张量赋值 / Test tensor assign"""
        x = paddle.to_tensor([1, 2, 3], dtype='float32')
        result = paddle.assign(x)
        np.testing.assert_array_equal(
            result.numpy(), np.array([1, 2, 3], dtype='float32')
        )

    def test_assign_with_output(self):
        """测试指定输出张量 / Test assign with output tensor"""
        output = paddle.empty(shape=[3], dtype='int64')
        result = paddle.assign([10, 20, 30], output)
        np.testing.assert_array_equal(result.numpy(), np.array([10, 20, 30]))


class TestToTensor(unittest.TestCase):
    """to_tensor 测试 / to_tensor tests"""

    def setUp(self):
        """测试前准备 / Setup before tests"""
        paddle.disable_static()

    def tearDown(self):
        """测试后清理 / Teardown after tests"""
        pass

    def test_to_tensor_list(self):
        """测试列表转张量 / Test list to tensor"""
        t = paddle.to_tensor([1, 2, 3])
        np.testing.assert_array_equal(t.numpy(), np.array([1, 2, 3]))

    def test_to_tensor_tuple(self):
        """测试元组转张量 / Test tuple to tensor"""
        t = paddle.to_tensor((4.0, 5.0, 6.0))
        np.testing.assert_array_equal(t.numpy(), np.array([4.0, 5.0, 6.0]))

    def test_to_tensor_ndarray(self):
        """测试 numpy 数组转张量 / Test numpy array to tensor"""
        data = np.array([[1, 2], [3, 4]], dtype='float32')
        t = paddle.to_tensor(data)
        np.testing.assert_array_equal(t.numpy(), data)

    def test_to_tensor_scalar(self):
        """测试标量转张量 / Test scalar to tensor"""
        t = paddle.to_tensor(3.14)
        self.assertAlmostEqual(t.item(), 3.14, places=5)

    def test_to_tensor_with_dtype(self):
        """测试指定数据类型转张量 / Test to_tensor with dtype"""
        t = paddle.to_tensor([1, 2, 3], dtype='float64')
        self.assertEqual(t.dtype, paddle.float64)

    def test_to_tensor_bool(self):
        """测试布尔值转张量 / Test bool to tensor"""
        t = paddle.to_tensor(True)
        self.assertEqual(t.item(), True)

    def test_to_tensor_nested_list(self):
        """测试嵌套列表转张量 / Test nested list to tensor"""
        t = paddle.to_tensor([[1, 2], [3, 4]])
        self.assertEqual(t.shape, [2, 2])

    def test_to_tensor_stop_gradient_false(self):
        """测试 stop_gradient 参数 / Test stop_gradient parameter"""
        t = paddle.to_tensor([1.0, 2.0], dtype='float32', stop_gradient=False)
        self.assertFalse(t.stop_gradient)


class TestFullAndZerosOnes(unittest.TestCase):
    """full / zeros / ones 创建测试 / full/zeros/ones creation tests"""

    def setUp(self):
        """测试前准备 / Setup before tests"""
        paddle.disable_static()

    def tearDown(self):
        """测试后清理 / Teardown after tests"""
        pass

    def test_full_basic(self):
        """测试基本 full / Test basic full"""
        result = paddle.full(shape=[2, 3], fill_value=5.0, dtype='float32')
        expected = np.full((2, 3), 5.0, dtype='float32')
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_full_int(self):
        """测试整数 full / Test int full"""
        result = paddle.full(shape=[3, 3], fill_value=7, dtype='int64')
        expected = np.full((3, 3), 7, dtype='int64')
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_full_bool(self):
        """测试布尔 full / Test bool full"""
        result = paddle.full(shape=[2, 2], fill_value=True, dtype='bool')
        expected = np.full((2, 2), True, dtype='bool')
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_full_negative_value(self):
        """测试负值 full / Test negative value full"""
        result = paddle.full(shape=[2, 2], fill_value=-1.5, dtype='float32')
        expected = np.full((2, 2), -1.5, dtype='float32')
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_zeros_basic(self):
        """测试基本 zeros / Test basic zeros"""
        result = paddle.zeros(shape=[3, 4], dtype='float32')
        expected = np.zeros((3, 4), dtype='float32')
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_ones_basic(self):
        """测试基本 ones / Test basic ones"""
        result = paddle.ones(shape=[2, 3], dtype='float32')
        expected = np.ones((2, 3), dtype='float32')
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_ones_int(self):
        """测试整数 ones / Test int ones"""
        result = paddle.ones(shape=[2, 3], dtype='int64')
        expected = np.ones((2, 3), dtype='int64')
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_zeros_like_basic(self):
        """测试 zeros_like / Test zeros_like"""
        x = paddle.ones(shape=[3, 4], dtype='float32')
        result = paddle.zeros_like(x)
        expected = np.zeros((3, 4), dtype='float32')
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_ones_like_basic(self):
        """测试 ones_like / Test ones_like"""
        x = paddle.zeros(shape=[3, 4], dtype='float32')
        result = paddle.ones_like(x)
        expected = np.ones((3, 4), dtype='float32')
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_full_like_basic(self):
        """测试 full_like / Test full_like"""
        x = paddle.zeros(shape=[3, 4], dtype='float32')
        result = paddle.full_like(x, fill_value=7.0)
        expected = np.full((3, 4), 7.0, dtype='float32')
        np.testing.assert_array_equal(result.numpy(), expected)


class TestMeshgrid(unittest.TestCase):
    """meshgrid 测试 / meshgrid tests"""

    def setUp(self):
        """测试前准备 / Setup before tests"""
        paddle.disable_static()

    def tearDown(self):
        """测试后清理 / Teardown after tests"""
        pass

    def test_meshgrid_basic(self):
        """测试基本 meshgrid / Test basic meshgrid"""
        x = paddle.to_tensor([1, 2, 3])
        y = paddle.to_tensor([4, 5])
        gx, gy = paddle.meshgrid(x, y)
        # 默认 'ij' indexing
        expected_gx, expected_gy = np.meshgrid([1, 2, 3], [4, 5], indexing='ij')
        np.testing.assert_array_equal(gx.numpy(), expected_gx)
        np.testing.assert_array_equal(gy.numpy(), expected_gy)

    def test_meshgrid_xy_indexing(self):
        """测试 xy indexing meshgrid / Test xy indexing meshgrid"""
        x = paddle.to_tensor([1, 2, 3])
        y = paddle.to_tensor([4, 5])
        gx, gy = paddle.meshgrid(x, y, indexing='xy')
        expected_gx, expected_gy = np.meshgrid([1, 2, 3], [4, 5], indexing='xy')
        np.testing.assert_array_equal(gx.numpy(), expected_gx)
        np.testing.assert_array_equal(gy.numpy(), expected_gy)

    def test_meshgrid_single_input(self):
        """测试单输入 meshgrid / Test single input meshgrid"""
        x = paddle.to_tensor([1, 2, 3])
        result = paddle.meshgrid(x)
        self.assertEqual(len(result), 1)
        np.testing.assert_array_equal(result[0].numpy(), np.array([1, 2, 3]))


class TestEye(unittest.TestCase):
    """eye 测试 / eye tests"""

    def setUp(self):
        """测试前准备 / Setup before tests"""
        paddle.disable_static()

    def tearDown(self):
        """测试后清理 / Teardown after tests"""
        pass

    def test_eye_basic(self):
        """测试基本 eye / Test basic eye"""
        result = paddle.eye(3, dtype='float32')
        expected = np.eye(3, dtype='float32')
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_eye_rectangular(self):
        """测试矩形 eye / Test rectangular eye"""
        result = paddle.eye(3, 4, dtype='float32')
        expected = np.eye(3, 4, dtype='float32')
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_eye_with_offset(self):
        """测试非方阵 eye / Test rectangular eye"""
        result = paddle.eye(3, 5, dtype='float32')
        expected = np.eye(3, 5, dtype='float32')
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_eye_int64(self):
        """测试 int64 eye / Test int64 eye"""
        result = paddle.eye(3, dtype='int64')
        self.assertEqual(result.dtype, paddle.int64)


if __name__ == '__main__':
    unittest.main()
