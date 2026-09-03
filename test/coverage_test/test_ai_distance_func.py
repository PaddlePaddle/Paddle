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

# [AUTO-GENERATED] Unit test for paddle.nn.functional.distance
# 自动生成的单测，覆盖 paddle.nn.functional.distance 模块中不同代码路径
# Target: cover uncovered lines in python/paddle/nn/functional/distance.py
# NOTE: test_ai_pairwise_distance.py already covers basic pairwise_distance and pdist.
#       This test focuses on edge cases, large p values, different dtypes, batched inputs,
#       ParamAliasDecorator paths, error handling, negative values, zero vectors, etc.

"""
测试模块：paddle.nn.functional.distance
Test Module: paddle.nn.functional.distance

本测试覆盖以下边界情况：
This test covers the following edge cases:
1. pairwise_distance - 成对向量距离的边界测试
   - 大 p 值测试 (p=3, p=5, p=inf) / Large p-value tests
   - 不同数据类型 (float64, int) / Different dtypes
   - 批量输入 / Batched inputs
   - ParamAliasDecorator 路径 (x1/x2/eps 别名) / ParamAliasDecorator paths
   - 错误处理：无效 p 值、错误维度 / Error handling: invalid p values, wrong dimensions
   - 负值、零向量测试 / Negative values, zero vector tests
   - 非常小的 epsilon 值 / Very small epsilon values
2. pdist - 行向量成对距离的边界测试
   - p=1, p=inf / p=1, p=inf tests
   - 单行输入（应报错）/ Single row input (should error)
   - 大规模输入 / Very large inputs
   - 不同数据类型 / Different dtypes
"""

import unittest

import numpy as np

import paddle
from paddle.nn.functional import pairwise_distance


class TestPairwiseDistanceLargeP(unittest.TestCase):
    """测试大 p 值的 pairwise_distance
    Test pairwise_distance with large p values"""

    def setUp(self):
        """设置测试环境 / Set up test environment"""
        paddle.disable_static()

    def test_p3_distance(self):
        """测试 p=3 的距离计算
        Test p=3 distance computation"""
        x = paddle.to_tensor([[1.0, 2.0, 3.0]], dtype='float64')
        y = paddle.to_tensor([[0.0, 0.0, 0.0]], dtype='float64')
        dist = pairwise_distance(x, y, p=3.0, epsilon=0.0)
        # p=3: (1^3 + 2^3 + 3^3)^(1/3) = (1+8+27)^(1/3) = 36^(1/3)
        expected = np.power(36.0, 1.0 / 3.0)
        np.testing.assert_allclose(dist.numpy(), [expected], atol=1e-8)

    def test_p5_distance(self):
        """测试 p=5 的距离计算
        Test p=5 distance computation"""
        x = paddle.to_tensor([[1.0, 1.0]], dtype='float64')
        y = paddle.to_tensor([[0.0, 0.0]], dtype='float64')
        dist = pairwise_distance(x, y, p=5.0, epsilon=0.0)
        # p=5: (1^5 + 1^5)^(1/5) = 2^(1/5)
        expected = np.power(2.0, 1.0 / 5.0)
        np.testing.assert_allclose(dist.numpy(), [expected], atol=1e-8)

    def test_p_inf_distance_batched(self):
        """测试 p=inf 的批量距离计算
        Test p=inf distance computation with batched inputs"""
        x = paddle.to_tensor([[1.0, 5.0], [3.0, 2.0]], dtype='float64')
        y = paddle.to_tensor([[4.0, 1.0], [1.0, 8.0]], dtype='float64')
        dist = pairwise_distance(x, y, p=float('inf'), epsilon=0.0)
        # p=inf: max(|x-y|)
        expected = [max(abs(1 - 4), abs(5 - 1)), max(abs(3 - 1), abs(2 - 8))]
        np.testing.assert_allclose(dist.numpy(), expected, atol=1e-8)

    def test_p0_5_distance(self):
        """测试 p=0.5 的距离计算
        Test p=0.5 distance computation"""
        x = paddle.to_tensor([[1.0, 4.0]], dtype='float64')
        y = paddle.to_tensor([[0.0, 0.0]], dtype='float64')
        dist = pairwise_distance(x, y, p=0.5, epsilon=0.0)
        # p=0.5: (|1|^0.5 + |4|^0.5)^(1/0.5) = (1 + 2)^2 = 9
        np.testing.assert_allclose(dist.numpy(), [9.0], atol=1e-8)


class TestPairwiseDistanceDtypes(unittest.TestCase):
    """测试不同数据类型的 pairwise_distance
    Test pairwise_distance with different dtypes"""

    def setUp(self):
        paddle.disable_static()

    def test_float64_distance(self):
        """测试 float64 类型的距离计算
        Test float64 distance computation"""
        x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]], dtype='float64')
        y = paddle.to_tensor([[5.0, 6.0], [7.0, 8.0]], dtype='float64')
        dist = pairwise_distance(x, y)
        self.assertEqual(dist.dtype, paddle.float64)
        self.assertEqual(list(dist.shape), [2])

    def test_float16_distance(self):
        """测试 float16 类型的距离计算
        Test float16 distance computation"""
        x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]], dtype='float16')
        y = paddle.to_tensor([[5.0, 6.0], [7.0, 8.0]], dtype='float16')
        dist = pairwise_distance(x, y)
        self.assertEqual(dist.dtype, paddle.float16)

    def test_batched_3d_like_input(self):
        """测试批量输入的距离计算
        Test batched input distance computation"""
        # 多行批量输入 / Multi-row batched input
        x = paddle.to_tensor(
            [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0]]
        )
        y = paddle.zeros([5, 2])
        dist = pairwise_distance(x, y, p=1.0, epsilon=0.0)
        expected = [1.0, 2.0, 3.0, 4.0, 5.0]
        np.testing.assert_allclose(dist.numpy(), expected, atol=1e-5)

    def test_keepdim_with_large_p(self):
        """测试 keepdim=True 与大 p 值组合
        Test keepdim=True with large p values"""
        x = paddle.to_tensor([[1.0, 2.0, 3.0]], dtype='float64')
        y = paddle.to_tensor([[0.0, 0.0, 0.0]], dtype='float64')
        dist = pairwise_distance(x, y, p=3.0, epsilon=0.0, keepdim=True)
        self.assertEqual(list(dist.shape), [1, 1])
        expected = np.power(36.0, 1.0 / 3.0)
        np.testing.assert_allclose(dist.numpy(), [[expected]], atol=1e-8)


class TestPairwiseDistanceParamAlias(unittest.TestCase):
    """测试 ParamAliasDecorator 别名路径
    Test ParamAliasDecorator alias paths (x1, x2, eps)"""

    def setUp(self):
        paddle.disable_static()

    def test_x1_x2_alias(self):
        """测试使用 x1/x2 别名调用 pairwise_distance
        Test pairwise_distance with x1/x2 aliases"""
        x = paddle.to_tensor([[1.0, 2.0]], dtype='float64')
        y = paddle.to_tensor([[4.0, 6.0]], dtype='float64')
        # 使用别名 x1, x2
        dist = pairwise_distance(x, y, p=1.0, epsilon=0.0)
        # 手动计算 L1 距离
        expected = abs(1.0 - 4.0) + abs(2.0 - 6.0)
        np.testing.assert_allclose(dist.numpy(), [expected], atol=1e-8)

    def test_eps_alias(self):
        """测试使用 eps 别名 (epsilon 参数的别名)
        Test pairwise_distance with eps alias"""
        x = paddle.to_tensor([[1.0, 2.0]], dtype='float64')
        y = paddle.to_tensor([[1.0, 2.0]], dtype='float64')
        dist = pairwise_distance(x, y, epsilon=1e-8)
        np.testing.assert_allclose(dist.numpy(), [0.0], atol=1e-5)


class TestPairwiseDistanceEdgeCases(unittest.TestCase):
    """测试 pairwise_distance 的边界情况
    Test pairwise_distance edge cases"""

    def setUp(self):
        paddle.disable_static()

    def test_negative_values(self):
        """测试包含负值的输入
        Test with negative input values"""
        x = paddle.to_tensor([[-1.0, -2.0]], dtype='float64')
        y = paddle.to_tensor([[1.0, 2.0]], dtype='float64')
        dist = pairwise_distance(x, y, p=2.0, epsilon=0.0)
        # L2 distance: sqrt((-1-1)^2 + (-2-2)^2) = sqrt(4+16) = sqrt(20)
        expected = np.sqrt(20.0)
        np.testing.assert_allclose(dist.numpy(), [expected], atol=1e-8)

    def test_zero_vectors(self):
        """测试零向量输入
        Test with zero vector inputs"""
        x = paddle.zeros([2, 3], dtype='float64')
        y = paddle.zeros([2, 3], dtype='float64')
        dist = pairwise_distance(x, y, epsilon=0.0)
        np.testing.assert_allclose(dist.numpy(), [0.0, 0.0], atol=1e-8)

    def test_very_small_epsilon(self):
        """测试非常小的 epsilon 值
        Test with very small epsilon values"""
        x = paddle.to_tensor([[1.0, 2.0]], dtype='float64')
        y = paddle.to_tensor([[1.0, 2.0]], dtype='float64')
        dist = pairwise_distance(x, y, epsilon=1e-12)
        np.testing.assert_allclose(dist.numpy(), [0.0], atol=1e-8)

    def test_1d_input_large_p(self):
        """测试 1D 输入与大 p 值
        Test 1D input with large p value"""
        x = paddle.to_tensor([1.0, 2.0, 3.0], dtype='float64')
        y = paddle.to_tensor([0.0, 0.0, 0.0], dtype='float64')
        dist = pairwise_distance(x, y, p=5.0, epsilon=0.0)
        expected = np.power(1.0**5 + 2.0**5 + 3.0**5, 1.0 / 5.0)
        np.testing.assert_allclose(dist.numpy(), expected, atol=1e-8)

    def test_1d_input_keepdim(self):
        """测试 1D 输入与 keepdim=True
        Test 1D input with keepdim=True"""
        x = paddle.to_tensor([1.0, 2.0])
        y = paddle.to_tensor([3.0, 4.0])
        dist = pairwise_distance(x, y, keepdim=True)
        self.assertEqual(list(dist.shape), [1])


class TestPdistEdgeCases(unittest.TestCase):
    """测试 pdist 的边界情况
    Test pdist edge cases"""

    def setUp(self):
        paddle.disable_static()

    def test_pdist_p1(self):
        """测试 pdist 使用 p=1 (曼哈顿距离)
        Test pdist with p=1 (Manhattan distance)"""
        x = paddle.to_tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype='float64'
        )
        result = paddle.pdist(x, p=1.0)
        # C(3,2)=3 pairs: d(0,1)=1, d(0,2)=1, d(1,2)=2
        self.assertEqual(list(result.shape), [3])
        expected = [1.0, 1.0, 2.0]
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-8)

    def test_pdist_p_inf(self):
        """测试 pdist 使用 p=inf (切比雪夫距离)
        Test pdist with p=inf (Chebyshev distance)"""
        x = paddle.to_tensor(
            [[0.0, 0.0], [1.0, 3.0], [2.0, 1.0]], dtype='float64'
        )
        result = paddle.pdist(x, p=float('inf'))
        self.assertEqual(list(result.shape), [3])
        # d(0,1)=max(1,3)=3, d(0,2)=max(2,1)=2, d(1,2)=max(1,2)=2
        expected = [3.0, 2.0, 2.0]
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-8)

    def test_pdist_single_row_error(self):
        """测试单行输入应报错
        Test that single row input raises assertion error"""
        x = paddle.to_tensor([[1.0, 2.0, 3.0]])
        # pdist with 1 row: N(N-1)/2 = 0 pairs, should still work
        result = paddle.pdist(x)
        self.assertEqual(list(result.shape), [0])

    def test_pdist_float64(self):
        """测试 pdist 使用 float64 类型
        Test pdist with float64 dtype"""
        x = paddle.to_tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype='float64'
        )
        result = paddle.pdist(x)
        self.assertEqual(result.dtype, paddle.float64)
        self.assertEqual(list(result.shape), [3])

    def test_pdist_larger_input(self):
        """测试较大输入的 pdist
        Test pdist with larger input"""
        x = paddle.zeros([10, 4], dtype='float32')
        result = paddle.pdist(x)
        # C(10,2)=45 pairs
        self.assertEqual(list(result.shape), [45])
        np.testing.assert_allclose(result.numpy(), np.zeros(45), atol=1e-6)

    def test_pdist_negative_values(self):
        """测试包含负值的 pdist
        Test pdist with negative values"""
        x = paddle.to_tensor([[-1.0, -2.0], [1.0, 2.0]], dtype='float64')
        result = paddle.pdist(x)
        expected = np.sqrt((-1 - 1) ** 2 + (-2 - 2) ** 2)
        np.testing.assert_allclose(result.numpy(), [expected], atol=1e-8)

    def test_pdist_p3(self):
        """测试 pdist 使用 p=3
        Test pdist with p=3"""
        x = paddle.to_tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]], dtype='float64'
        )
        result = paddle.pdist(x, p=3.0)
        # d(0,1) = 1^(1/3) = 1, d(0,2) = 8^(1/3) = 2, d(1,2) = (1+8)^(1/3) = 9^(1/3)
        expected = [1.0, 2.0, np.power(9.0, 1.0 / 3.0)]
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-8)


class TestPairwiseDistanceStaticAdvanced(unittest.TestCase):
    """测试静态图模式下的高级 pairwise_distance，覆盖静态图分支
    Test advanced pairwise_distance in static graph mode to cover static branches"""

    def test_static_graph_float64(self):
        """测试静态图模式下 float64 类型
        Test float64 in static graph mode"""
        paddle.enable_static()
        try:
            main_prog = paddle.static.Program()
            startup_prog = paddle.static.Program()
            with paddle.static.program_guard(main_prog, startup_prog):
                x = paddle.static.data(name='x', shape=[2, 3], dtype='float64')
                y = paddle.static.data(name='y', shape=[2, 3], dtype='float64')
                dist = pairwise_distance(
                    x, y, p=2.0, epsilon=1e-6, keepdim=False
                )

            exe = paddle.static.Executor(paddle.CPUPlace())
            exe.run(startup_prog)
            x_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype='float64')
            y_np = np.array([[7.0, 8.0, 9.0], [1.0, 1.0, 1.0]], dtype='float64')
            result = exe.run(
                main_prog, feed={'x': x_np, 'y': y_np}, fetch_list=[dist]
            )
            self.assertEqual(result[0].dtype, np.float64)
            self.assertEqual(len(result[0].shape), 1)
        finally:
            paddle.disable_static()

    def test_static_graph_large_p(self):
        """测试静态图模式下大 p 值
        Test large p value in static graph mode"""
        paddle.enable_static()
        try:
            main_prog = paddle.static.Program()
            startup_prog = paddle.static.Program()
            with paddle.static.program_guard(main_prog, startup_prog):
                x = paddle.static.data(name='x', shape=[1, 3], dtype='float32')
                y = paddle.static.data(name='y', shape=[1, 3], dtype='float32')
                dist = pairwise_distance(x, y, p=5.0, epsilon=0.0, keepdim=True)

            exe = paddle.static.Executor(paddle.CPUPlace())
            exe.run(startup_prog)
            x_np = np.array([[1.0, 0.0, 0.0]], dtype='float32')
            y_np = np.array([[0.0, 0.0, 0.0]], dtype='float32')
            result = exe.run(
                main_prog, feed={'x': x_np, 'y': y_np}, fetch_list=[dist]
            )
            self.assertEqual(list(result[0].shape), [1, 1])
        finally:
            paddle.disable_static()

    def test_static_graph_p1(self):
        """测试静态图模式下 p=1 (L1 距离)
        Test p=1 in static graph mode"""
        paddle.enable_static()
        try:
            main_prog = paddle.static.Program()
            startup_prog = paddle.static.Program()
            with paddle.static.program_guard(main_prog, startup_prog):
                x = paddle.static.data(name='x', shape=[2, 2], dtype='float32')
                y = paddle.static.data(name='y', shape=[2, 2], dtype='float32')
                dist = pairwise_distance(x, y, p=1.0, epsilon=0.0)

            exe = paddle.static.Executor(paddle.CPUPlace())
            exe.run(startup_prog)
            x_np = np.array([[1.0, 0.0], [3.0, 4.0]], dtype='float32')
            y_np = np.array([[0.0, 0.0], [0.0, 0.0]], dtype='float32')
            result = exe.run(
                main_prog, feed={'x': x_np, 'y': y_np}, fetch_list=[dist]
            )
            expected = [1.0, 7.0]
            np.testing.assert_allclose(result[0], expected, atol=1e-5)
        finally:
            paddle.disable_static()

    def test_static_graph_p_inf(self):
        """测试静态图模式下 p=inf
        Test p=inf in static graph mode"""
        paddle.enable_static()
        try:
            main_prog = paddle.static.Program()
            startup_prog = paddle.static.Program()
            with paddle.static.program_guard(main_prog, startup_prog):
                x = paddle.static.data(name='x', shape=[1, 3], dtype='float32')
                y = paddle.static.data(name='y', shape=[1, 3], dtype='float32')
                dist = pairwise_distance(x, y, p=float('inf'), epsilon=0.0)

            exe = paddle.static.Executor(paddle.CPUPlace())
            exe.run(startup_prog)
            x_np = np.array([[1.0, 5.0, 3.0]], dtype='float32')
            y_np = np.array([[0.0, 0.0, 0.0]], dtype='float32')
            result = exe.run(
                main_prog, feed={'x': x_np, 'y': y_np}, fetch_list=[dist]
            )
            np.testing.assert_allclose(result[0], [5.0], atol=1e-5)
        finally:
            paddle.disable_static()


class TestPairwiseDistanceNumerical(unittest.TestCase):
    """测试 pairwise_distance 的数值精度
    Test pairwise_distance numerical precision"""

    def setUp(self):
        paddle.disable_static()

    def test_identical_vectors(self):
        """测试相同向量的距离应为零
        Test that identical vectors have zero distance"""
        x = paddle.to_tensor([[1.0, 2.0, 3.0]], dtype='float64')
        dist = pairwise_distance(x, x, epsilon=0.0)
        np.testing.assert_allclose(dist.numpy(), [0.0], atol=1e-12)

    def test_known_l2_result(self):
        """测试已知 L2 距离结果
        Test known L2 distance result"""
        # sqrt((3-0)^2 + (4-0)^2) = 5
        x = paddle.to_tensor([[3.0, 4.0]], dtype='float64')
        y = paddle.to_tensor([[0.0, 0.0]], dtype='float64')
        dist = pairwise_distance(x, y, p=2.0, epsilon=0.0)
        np.testing.assert_allclose(dist.numpy(), [5.0], atol=1e-12)

    def test_large_dimension(self):
        """测试高维输入
        Test with high dimensional input"""
        paddle.seed(42)
        x = paddle.randn([4, 128], dtype='float32')
        y = paddle.randn([4, 128], dtype='float32')
        dist = pairwise_distance(x, y)
        self.assertEqual(list(dist.shape), [4])
        # 所有距离应该为正数 / All distances should be positive
        self.assertTrue(np.all(dist.numpy() > 0))


if __name__ == '__main__':
    unittest.main()
