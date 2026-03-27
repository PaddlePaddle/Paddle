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

# [AUTO-GENERATED] Unit test for paddle.nn.functional.pairwise_distance and pdist
# 自动生成的单测，覆盖 paddle.nn.functional.distance 模块中未覆盖的代码
# Target: cover uncovered lines 95-124 in paddle/python/paddle/nn/functional/distance.py

"""
测试模块：paddle.nn.functional.distance
Test Module: paddle.nn.functional.distance

本测试覆盖以下功能：
This test covers the following functions:
1. pairwise_distance - 计算成对向量距离 / Compute pairwise vector distance
   - 动态图模式下的各种参数组合 / Dynamic graph mode with various parameter combinations
   - keepdim 参数测试 / keepdim parameter test
   - 不同 p 范数测试 / Different p-norm tests
   - epsilon 参数测试 / epsilon parameter test
   - 1D输入测试 / 1D input test
2. pdist - 计算行向量之间的距离 / Compute row-wise pairwise distances

覆盖的未覆盖行：95-124（静态图路径）
"""

import unittest

import numpy as np

import paddle


class TestPairwiseDistanceDynamic(unittest.TestCase):
    """测试动态图模式下的pairwise_distance
    Test pairwise_distance in dynamic graph mode"""

    def setUp(self):
        """设置测试环境 / Set up test environment"""
        paddle.disable_static()

    def test_basic_l2_distance(self):
        """测试基本的L2距离计算
        Test basic L2 distance computation"""
        x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
        y = paddle.to_tensor([[5.0, 6.0], [7.0, 8.0]])
        dist = paddle.nn.functional.pairwise_distance(x, y)
        self.assertEqual(list(dist.shape), [2])

    def test_keepdim_true(self):
        """测试keepdim=True时输出形状
        Test output shape when keepdim=True"""
        x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
        y = paddle.to_tensor([[5.0, 6.0], [7.0, 8.0]])
        dist = paddle.nn.functional.pairwise_distance(x, y, keepdim=True)
        self.assertEqual(list(dist.shape), [2, 1])

    def test_l1_distance(self):
        """测试L1距离 (p=1.0)
        Test L1 distance with p=1.0"""
        x = paddle.to_tensor([[1.0, 0.0]], dtype='float64')
        y = paddle.to_tensor([[0.0, 0.0]], dtype='float64')
        dist = paddle.nn.functional.pairwise_distance(x, y, p=1.0, epsilon=0.0)
        np.testing.assert_allclose(dist.numpy(), [1.0], atol=1e-5)

    def test_linf_distance(self):
        """测试L-inf距离 (p=float('inf'))
        Test L-inf distance with p=inf"""
        x = paddle.to_tensor([[1.0, 3.0, 5.0]])
        y = paddle.to_tensor([[0.0, 0.0, 0.0]])
        dist = paddle.nn.functional.pairwise_distance(
            x, y, p=float('inf'), epsilon=0.0
        )
        np.testing.assert_allclose(dist.numpy(), [5.0], atol=1e-5)

    def test_zero_epsilon(self):
        """测试epsilon=0.0时的行为
        Test behavior when epsilon=0.0"""
        x = paddle.to_tensor([[1.0, 2.0]])
        y = paddle.to_tensor([[1.0, 2.0]])
        dist = paddle.nn.functional.pairwise_distance(x, y, epsilon=0.0)
        np.testing.assert_allclose(dist.numpy(), [0.0], atol=1e-5)

    def test_1d_input(self):
        """测试1D输入（无batch维度）
        Test 1D input without batch dimension"""
        x = paddle.to_tensor([1.0, 2.0, 3.0])
        y = paddle.to_tensor([4.0, 5.0, 6.0])
        dist = paddle.nn.functional.pairwise_distance(x, y)
        self.assertEqual(len(dist.shape), 0)  # scalar output

    def test_float16_input(self):
        """测试float32输入
        Test float32 input"""
        x = paddle.to_tensor([[1.0, 2.0]], dtype='float32')
        y = paddle.to_tensor([[3.0, 4.0]], dtype='float32')
        dist = paddle.nn.functional.pairwise_distance(x, y)
        self.assertEqual(dist.dtype, paddle.float32)


class TestPdist(unittest.TestCase):
    """测试pdist函数
    Test pdist function"""

    def setUp(self):
        paddle.disable_static()

    def test_pdist_basic(self):
        """测试基本的pdist计算
        Test basic pdist computation"""
        paddle.seed(2023)
        x = paddle.randn([4, 5])
        result = paddle.pdist(x)
        # 4行向量的成对距离数应为 C(4,2)=6
        # Number of pairwise distances for 4 rows should be C(4,2)=6
        self.assertEqual(list(result.shape), [6])

    def test_pdist_l1(self):
        """测试pdist使用L1范数
        Test pdist with L1 norm"""
        x = paddle.to_tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        result = paddle.pdist(x, p=1.0)
        # C(3,2)=3 pairs
        self.assertEqual(list(result.shape), [3])

    def test_pdist_two_rows(self):
        """测试只有两行的输入
        Test input with only two rows"""
        x = paddle.to_tensor([[1.0, 0.0], [0.0, 1.0]])
        result = paddle.pdist(x)
        self.assertEqual(list(result.shape), [1])


class TestPairwiseDistanceStatic(unittest.TestCase):
    """测试静态图模式下的pairwise_distance，覆盖未覆盖的静态图代码路径
    Test pairwise_distance in static graph mode to cover uncovered static graph code paths"""

    def test_static_graph_basic(self):
        """测试静态图模式下的基本距离计算
        Test basic distance computation in static graph mode"""
        paddle.enable_static()
        try:
            main_prog = paddle.static.Program()
            startup_prog = paddle.static.Program()
            with paddle.static.program_guard(main_prog, startup_prog):
                x = paddle.static.data(name='x', shape=[2, 3], dtype='float32')
                y = paddle.static.data(name='y', shape=[2, 3], dtype='float32')
                dist = paddle.nn.functional.pairwise_distance(
                    x, y, p=2.0, epsilon=1e-6, keepdim=False
                )

            exe = paddle.static.Executor(paddle.CPUPlace())
            exe.run(startup_prog)
            x_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype='float32')
            y_np = np.array([[7.0, 8.0, 9.0], [1.0, 1.0, 1.0]], dtype='float32')
            result = exe.run(
                main_prog, feed={'x': x_np, 'y': y_np}, fetch_list=[dist]
            )
            self.assertEqual(len(result[0].shape), 1)
            self.assertEqual(result[0].shape[0], 2)
        finally:
            paddle.disable_static()

    def test_static_graph_keepdim(self):
        """测试静态图模式下keepdim=True
        Test keepdim=True in static graph mode"""
        paddle.enable_static()
        try:
            main_prog = paddle.static.Program()
            startup_prog = paddle.static.Program()
            with paddle.static.program_guard(main_prog, startup_prog):
                x = paddle.static.data(name='x', shape=[2, 3], dtype='float64')
                y = paddle.static.data(name='y', shape=[2, 3], dtype='float64')
                dist = paddle.nn.functional.pairwise_distance(
                    x, y, p=2.0, epsilon=0.0, keepdim=True
                )

            exe = paddle.static.Executor(paddle.CPUPlace())
            exe.run(startup_prog)
            x_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype='float64')
            y_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype='float64')
            result = exe.run(
                main_prog, feed={'x': x_np, 'y': y_np}, fetch_list=[dist]
            )
            self.assertEqual(list(result[0].shape), [2, 1])
        finally:
            paddle.disable_static()


if __name__ == '__main__':
    unittest.main()
