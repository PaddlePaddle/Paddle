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

# [AUTO-GENERATED] Tests for paddle/tensor/einsum.py (coverage: 94.8% -> higher)
# Target file: python/paddle/tensor/einsum.py
# Functions: einsum, parse_op_labels, parse_labels, validate_rhs, build_view,
#            build_global_view, build_global_shape, diagonalize, plan_einsum,
#            rhs_inference, gen_equation_for_opteinsum, replace_ellipsis

import os
import unittest

import numpy as np

import paddle
from paddle.tensor.einsum import (
    Plan,
    build_global_shape,
    build_view,
    diagonalize,
    gen_equation_for_opteinsum,
    has_duplicated_labels,
    parse_labels,
    parse_op_labels,
    rhs_inference,
    validate_rhs,
)


class TestEinsumBasic(unittest.TestCase):
    """测试 einsum 基本功能 / Test basic einsum functionality."""

    def setUp(self):
        paddle.seed(102)

    def tearDown(self):
        # Restore default
        if 'FLAGS_new_einsum' in os.environ:
            del os.environ['FLAGS_new_einsum']

    def test_einsum_sum(self):
        """测试 einsum 求和 / Test einsum sum."""
        x = paddle.rand([4])
        out = paddle.einsum('i->', x)
        np.testing.assert_allclose(out.numpy(), x.numpy().sum(), rtol=1e-5)

    def test_einsum_dot(self):
        """测试 einsum 点积 / Test einsum dot product."""
        x = paddle.rand([4])
        out = paddle.einsum('i,i->', x, x)
        expected = np.dot(x.numpy(), x.numpy())
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5)

    def test_einsum_outer(self):
        """测试 einsum 外积 / Test einsum outer product."""
        x = paddle.rand([4])
        y = paddle.rand([5])
        out = paddle.einsum('i,j->ij', x, y)
        self.assertEqual(out.shape, [4, 5])

    def test_einsum_transpose(self):
        """测试 einsum 转置 / Test einsum transpose."""
        x = paddle.rand([2, 3, 2])
        out = paddle.einsum('ijk->kji', x)
        self.assertEqual(out.shape, [2, 3, 2])

    def test_einsum_matmul(self):
        """测试 einsum 矩阵乘法 / Test einsum matrix multiplication."""
        a = paddle.rand([2, 3, 2])
        b = paddle.rand([2, 2, 3])
        out = paddle.einsum('ijk, ikl->ijl', a, b)
        self.assertEqual(out.shape, [2, 3, 3])

    def test_einsum_batch_matmul_ellipsis(self):
        """测试 einsum 批量矩阵乘法（省略号）/ Test einsum batch matmul with ellipsis."""
        a = paddle.rand([2, 3, 2])
        b = paddle.rand([2, 2, 3])
        out = paddle.einsum('...jk, ...kl->...jl', a, b)
        self.assertEqual(out.shape, [2, 3, 3])

    def test_einsum_elementwise(self):
        """测试 einsum 逐元素乘法 / Test einsum elementwise multiply."""
        a = paddle.rand([2, 3])
        b = paddle.rand([2, 3])
        out = paddle.einsum('ij,ij->ij', a, b)
        self.assertEqual(out.shape, [2, 3])

    def test_einsum_trace(self):
        """测试 einsum 迹 / Test einsum trace."""
        x = paddle.rand([3, 3])
        out = paddle.einsum('ii->', x)
        expected = np.trace(x.numpy())
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5)

    def test_einsum_diagonal(self):
        """测试 einsum 对角线 / Test einsum diagonal."""
        x = paddle.rand([3, 3])
        out = paddle.einsum('ii->i', x)
        self.assertEqual(out.shape, [3])

    def test_einsum_ellipsis_transpose(self):
        """测试 einsum 省略号转置 / Test einsum ellipsis transpose."""
        a = paddle.rand([2, 2, 3])
        out = paddle.einsum('...jk->...kj', a)
        self.assertEqual(out.shape, [2, 3, 2])

    def test_einsum_empty_output(self):
        """测试 einsum 空输出 / Test einsum with empty output (scalar)."""
        x = paddle.rand([2, 3])
        out = paddle.einsum('ij->', x)
        np.testing.assert_allclose(out.numpy(), x.numpy().sum(), rtol=1e-5)

    def test_einsum_inferred_output(self):
        """测试 einsum 推断输出 / Test einsum with inferred output."""
        x = paddle.rand([2, 3])
        y = paddle.rand([3, 4])
        out = paddle.einsum('ij,jk', x, y)
        self.assertEqual(out.shape, [2, 4])


class TestEinsumMultiOperand(unittest.TestCase):
    """测试 einsum 多操作数 / Test einsum with multiple operands."""

    def setUp(self):
        paddle.seed(42)

    def tearDown(self):
        if 'FLAGS_new_einsum' in os.environ:
            del os.environ['FLAGS_new_einsum']

    def test_einsum_three_operands(self):
        """测试 einsum 三操作数 / Test einsum with three operands."""
        a = paddle.rand([2, 3])
        b = paddle.rand([3, 4])
        c = paddle.rand([4, 2])
        out = paddle.einsum('ij,jk,ki->', a, b, c)
        self.assertEqual(out.shape, [])


class TestEinsumV2(unittest.TestCase):
    """测试 einsum_v2 (新实现) / Test einsum_v2 (new implementation)."""

    def setUp(self):
        os.environ['FLAGS_new_einsum'] = '1'
        paddle.seed(102)

    def tearDown(self):
        if 'FLAGS_new_einsum' in os.environ:
            del os.environ['FLAGS_new_einsum']

    def test_einsum_v2_matmul(self):
        """测试 einsum_v2 矩阵乘法 / Test einsum_v2 matrix multiplication."""
        a = paddle.rand([2, 3, 2])
        b = paddle.rand([2, 2, 3])
        out = paddle.einsum('ijk, ikl->ijl', a, b)
        self.assertEqual(out.shape, [2, 3, 3])

    def test_einsum_v2_ellipsis(self):
        """测试 einsum_v2 省略号 / Test einsum_v2 with ellipsis."""
        a = paddle.rand([2, 3, 2])
        b = paddle.rand([2, 2, 3])
        out = paddle.einsum('...jk, ...kl->...jl', a, b)
        self.assertEqual(out.shape, [2, 3, 3])


class TestParseOpLabels(unittest.TestCase):
    """测试 parse_op_labels 功能 / Test parse_op_labels functionality."""

    def test_parse_op_labels_simple(self):
        """测试简单标签解析 / Test simple label parsing."""
        x = paddle.rand([2, 3])
        out = parse_op_labels('ij', x)
        self.assertEqual(out, 'ij')

    def test_parse_op_labels_ellipsis(self):
        """测试省略号标签解析 / Test ellipsis label parsing."""
        x = paddle.rand([2, 3, 4, 5])
        # parse_op_labels expands '...' to dots matching tensor dimensions
        out = parse_op_labels('ij...k', x)
        # 2D tensor labels 'ij' + 1 broadcast dim '.' + 'k' = 4 chars
        self.assertEqual(len(out), 4)
        self.assertTrue('.' in out)


class TestParseLabels(unittest.TestCase):
    """测试 parse_labels 功能 / Test parse_labels functionality."""

    def test_parse_labels_single(self):
        """测试单操作数标签解析 / Test single operand label parsing."""
        x = paddle.rand([2, 3])
        out = parse_labels('ij', [x])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0], 'ij')

    def test_parse_labels_multi(self):
        """测试多操作数标签解析 / Test multi operand label parsing."""
        x = paddle.rand([2, 3])
        y = paddle.rand([3, 4])
        out = parse_labels('ij,jk', [x, y])
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0], 'ij')
        self.assertEqual(out[1], 'jk')


class TestValidateRhs(unittest.TestCase):
    """测试 validate_rhs 功能 / Test validate_rhs functionality."""

    def test_validate_rhs_basic(self):
        """测试 validate_rhs 基本功能 / Test basic validate_rhs."""
        # Should not raise
        validate_rhs('ij', ['i', 'j'], n_bcast_dims=0)

    def test_validate_rhs_with_bcast(self):
        """测试 validate_rhs 带广播 / Test validate_rhs with broadcast."""
        validate_rhs('ij...', ['i', 'j'], n_bcast_dims=1)

    def test_validate_rhs_duplicate(self):
        """测试 validate_rhs 重复标签 / Test validate_rhs with duplicate labels."""
        with self.assertRaises(AssertionError):
            validate_rhs('ii', ['i'], n_bcast_dims=0)

    def test_validate_rhs_unknown_label(self):
        """测试 validate_rhs 未知标签 / Test validate_rhs with unknown label."""
        with self.assertRaises(AssertionError):
            validate_rhs('k', ['i', 'j'], n_bcast_dims=0)


class TestBuildView(unittest.TestCase):
    """测试 build_view 功能 / Test build_view functionality."""

    def test_build_view_basic(self):
        """测试 build_view 基本功能 / Test basic build_view."""
        result = build_view('ij', 'ji')
        self.assertEqual(result, [1, 0])

    def test_build_view_with_broadcast(self):
        """测试 build_view 带广播 / Test build_view with broadcast."""
        result = build_view('ij..', '..ji')
        self.assertEqual(result, [2, 3, 1, 0])

    def test_build_view_extra_dims(self):
        """测试 build_view 额外维度 / Test build_view with extra dims."""
        result = build_view('ij..', '..kji')
        # ..kji has 5 chars, ij.. has 4 chars -> 'k' has no match -> -1
        # From docstring: inv_map = [2, 3, -1, 1, 0]
        self.assertEqual(result[0], 2)
        self.assertEqual(result[1], 3)
        self.assertEqual(result[2], -1)
        self.assertEqual(result[3], 1)
        self.assertEqual(result[4], 0)


class TestBuildGlobalShape(unittest.TestCase):
    """测试 build_global_shape 功能 / Test build_global_shape functionality."""

    def test_build_global_shape_basic(self):
        """测试 build_global_shape 基本功能 / Test basic build_global_shape."""
        # 2 operands with matching shape [2, 3]
        g_view = [[0, 1], [0, 1]]
        g_labels = 'ij'
        op_shapes = [[2, 3], [2, 3]]
        g_shape, g_masks = build_global_shape(g_view, g_labels, op_shapes)
        self.assertEqual(g_shape, [2, 3])


class TestHasDuplicatedLabels(unittest.TestCase):
    """测试 has_duplicated_labels 功能 / Test has_duplicated_labels functionality."""

    def test_has_duplicated_true(self):
        """测试有重复标签 / Test with duplicated labels."""
        self.assertTrue(has_duplicated_labels('iij'))

    def test_has_duplicated_false(self):
        """测试无重复标签 / Test without duplicated labels."""
        self.assertFalse(has_duplicated_labels('ijk'))


class TestDiagonalize(unittest.TestCase):
    """测试 diagonalize 功能 / Test diagonalize functionality."""

    def test_diagonalize_basic(self):
        """测试 diagonalize 基本功能 / Test basic diagonalize."""
        x = paddle.rand([2, 3, 2])
        labels = 'ijk'
        new_labels, operand = diagonalize(labels, x)
        # No duplicated labels, so no change
        self.assertEqual(labels, new_labels)


class TestRhsInference(unittest.TestCase):
    """测试 rhs_inference 功能 / Test rhs_inference functionality."""

    def test_rhs_inference_basic(self):
        """测试 rhs_inference 基本功能 / Test basic rhs_inference."""
        result = rhs_inference('ij,jk')
        self.assertEqual(result, 'ik')

    def test_rhs_inference_with_ellipsis(self):
        """测试 rhs_inference 带省略号 / Test rhs_inference with ellipsis."""
        result = rhs_inference('...ij,...jk')
        self.assertTrue('...' in result)


class TestGenEquationForOpteinsum(unittest.TestCase):
    """测试 gen_equation_for_opteinsum 功能 / Test gen_equation_for_opteinsum functionality."""

    def test_gen_equation_basic(self):
        """测试 gen_equation 基本功能 / Test basic gen_equation."""
        eq, label = gen_equation_for_opteinsum('ij,jk', 'ik')
        self.assertTrue('->' in eq)

    def test_gen_equation_inferred_rhs(self):
        """测试 gen_equation 推断 rhs / Test gen_equation with inferred rhs."""
        eq, label = gen_equation_for_opteinsum('ij,jk', None)
        self.assertTrue('->' in eq)


class TestPlanClass(unittest.TestCase):
    """测试 Plan 类功能 / Test Plan class functionality."""

    def test_plan_basic(self):
        """测试 Plan 基本功能 / Test basic Plan."""
        plan = Plan()
        plan.set_var('x', paddle.to_tensor([1.0, 2.0]))
        out = plan.get_var('x')
        self.assertIsNotNone(out)

    def test_plan_execute(self):
        """测试 Plan 执行 / Test Plan execution."""
        plan = Plan()
        x = paddle.to_tensor([1.0, 2.0, 3.0])
        plan.set_var('x', x)
        plan.add_step((lambda v: v * 2, ['x'], 'y'))
        result = plan.execute()
        np.testing.assert_array_almost_equal(result.numpy(), [2.0, 4.0, 6.0])


if __name__ == '__main__':
    unittest.main()
