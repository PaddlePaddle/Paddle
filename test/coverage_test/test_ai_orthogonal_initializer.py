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

# [AUTO-GENERATED] Unit test for paddle.nn.initializer.Orthogonal
# 自动生成的单测，覆盖 paddle.nn.initializer.orthogonal 模块中未覆盖的代码
# Target: cover uncovered lines 157-271 in paddle/python/paddle/nn/initializer/orthogonal.py

"""
测试模块：paddle.nn.initializer.Orthogonal
Test Module: paddle.nn.initializer.Orthogonal

本测试覆盖以下功能：
This test covers the following functions:
1. Orthogonal 初始化器在动态图模式下的使用 / Orthogonal initializer in dynamic graph mode
   - rows > cols 情况：列向量正交 / rows > cols case: columns are orthogonal
   - rows < cols 情况：行向量正交 / rows < cols case: rows are orthogonal
   - rows == cols 情况：行列均正交 / rows == cols case: both rows and columns are orthogonal
   - gain 参数测试 / gain parameter test
2. Orthogonal 初始化器在静态图模式下的使用 / Orthogonal initializer in static graph mode
   - 覆盖静态图代码路径 lines 157-271 / covers static graph code path

覆盖的未覆盖行：157, 161, 167, 181, 187, 193, 204, 210, 218-219, 226, 234, 240-241, 247, 254, 256, 264, 271

注意：paddle.nn.Linear(in, out) 的weight形状为 [in, out]
Note: paddle.nn.Linear(in, out) weight shape is [in, out]
所以 Linear(10, 15) → weight [10, 15]，rows=10 < cols=15
"""

import unittest

import numpy as np

import paddle


class TestOrthogonalInitializerDynamic(unittest.TestCase):
    """测试动态图模式下Orthogonal初始化器
    Test Orthogonal initializer in dynamic graph mode"""

    def setUp(self):
        """设置测试环境 / Set up test environment"""
        paddle.disable_static()

    def test_rows_greater_than_cols(self):
        """测试rows > cols的情况，weight [15, 10]，列向量应正交
        Test rows > cols case, weight [15, 10], columns should be orthogonal
        Linear(10, 15) → weight shape [10, 15], rows=10 < cols=15
        So we use Linear(15, 10) → weight shape [15, 10], rows=15 > cols=10"""
        weight_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Orthogonal()
        )
        # Linear(out=10, in=15) → weight [15, 10], but Paddle Linear(in, out)
        # Linear(15, 10) → weight [15, 10], rows=15 > cols=10
        linear = paddle.nn.Linear(15, 10, weight_attr=weight_attr)
        weight = linear.weight.numpy()
        self.assertEqual(weight.shape, (15, 10))
        # W^T * W 应接近单位矩阵 / W^T * W should be close to identity (10x10)
        wtw = np.matmul(weight.T, weight)
        identity = np.eye(10)
        np.testing.assert_allclose(wtw, identity, atol=1e-5)

    def test_rows_less_than_cols(self):
        """测试rows < cols的情况，weight [10, 15]，行向量应正交
        Test rows < cols case, weight [10, 15], rows should be orthogonal
        Linear(10, 15) → weight [10, 15], rows=10 < cols=15"""
        weight_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Orthogonal()
        )
        linear = paddle.nn.Linear(10, 15, weight_attr=weight_attr)
        weight = linear.weight.numpy()
        self.assertEqual(weight.shape, (10, 15))
        # W * W^T 应接近单位矩阵 / W * W^T should be close to identity (10x10)
        wwt = np.matmul(weight, weight.T)
        identity = np.eye(10)
        np.testing.assert_allclose(wwt, identity, atol=1e-5)

    def test_square_matrix(self):
        """测试rows == cols的情况（10x10），行列均正交
        Test rows == cols case (10x10), both rows and columns are orthogonal"""
        weight_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Orthogonal()
        )
        linear = paddle.nn.Linear(10, 10, weight_attr=weight_attr)
        weight = linear.weight.numpy()
        self.assertEqual(weight.shape, (10, 10))
        # W * W^T 应接近单位矩阵 / W * W^T should be close to identity
        wwt = np.matmul(weight, weight.T)
        identity = np.eye(10)
        np.testing.assert_allclose(wwt, identity, atol=1e-5)

    def test_with_gain(self):
        """测试带gain参数的Orthogonal初始化
        Test Orthogonal initialization with gain parameter"""
        gain = 2.0
        weight_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Orthogonal(gain=gain)
        )
        linear = paddle.nn.Linear(10, 10, weight_attr=weight_attr)
        weight = linear.weight.numpy()
        # W * W^T / gain^2 应接近单位矩阵
        # W * W^T / gain^2 should be close to identity
        wwt = np.matmul(weight, weight.T) / (gain**2)
        identity = np.eye(10)
        np.testing.assert_allclose(wwt, identity, atol=1e-5)

    def test_conv2d_orthogonal(self):
        """测试在Conv2D上使用Orthogonal初始化（维度>2）
        Test Orthogonal initialization on Conv2D (dimension > 2)"""
        weight_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Orthogonal()
        )
        conv = paddle.nn.Conv2D(3, 8, 3, weight_attr=weight_attr)
        weight = conv.weight.numpy()
        self.assertEqual(len(weight.shape), 4)  # [out_c, in_c, kH, kW]


class TestOrthogonalInitializerStatic(unittest.TestCase):
    """测试静态图模式下的Orthogonal初始化器，覆盖未覆盖的静态图代码路径
    Test Orthogonal initializer in static graph mode, covering uncovered static paths"""

    def test_static_graph_rows_greater_cols(self):
        """测试静态图模式下rows > cols的Orthogonal初始化
        Test Orthogonal init with rows > cols in static graph mode
        Linear(15, 8) → weight [15, 8], rows=15 > cols=8"""
        paddle.enable_static()
        try:
            main_prog = paddle.static.Program()
            startup_prog = paddle.static.Program()
            with paddle.static.program_guard(main_prog, startup_prog):
                weight_attr = paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Orthogonal()
                )
                linear = paddle.nn.Linear(15, 8, weight_attr=weight_attr)

            exe = paddle.static.Executor(paddle.CPUPlace())
            exe.run(startup_prog)
            result = exe.run(main_prog, fetch_list=[linear.weight])
            weight = result[0]
            self.assertEqual(weight.shape, (15, 8))
            # W^T * W 应接近单位矩阵 / W^T * W should be close to identity (8x8)
            wtw = np.matmul(weight.T, weight)
            identity = np.eye(8)
            np.testing.assert_allclose(wtw, identity, atol=1e-4)
        finally:
            paddle.disable_static()

    def test_static_graph_rows_less_cols(self):
        """测试静态图模式下rows < cols的Orthogonal初始化（触发transpose分支）
        Test Orthogonal init with rows < cols in static graph (triggers transpose branch)
        Linear(8, 15) → weight [8, 15], rows=8 < cols=15"""
        paddle.enable_static()
        try:
            main_prog = paddle.static.Program()
            startup_prog = paddle.static.Program()
            with paddle.static.program_guard(main_prog, startup_prog):
                weight_attr = paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Orthogonal()
                )
                linear = paddle.nn.Linear(8, 15, weight_attr=weight_attr)

            exe = paddle.static.Executor(paddle.CPUPlace())
            exe.run(startup_prog)
            result = exe.run(main_prog, fetch_list=[linear.weight])
            weight = result[0]
            self.assertEqual(weight.shape, (8, 15))
            # W * W^T 应接近单位矩阵 / W * W^T should be close to identity (8x8)
            wwt = np.matmul(weight, weight.T)
            identity = np.eye(8)
            np.testing.assert_allclose(wwt, identity, atol=1e-4)
        finally:
            paddle.disable_static()

    def test_static_graph_with_gain(self):
        """测试静态图模式下带gain参数的Orthogonal初始化
        Test Orthogonal init with gain parameter in static graph mode"""
        paddle.enable_static()
        try:
            gain = 1.5
            main_prog = paddle.static.Program()
            startup_prog = paddle.static.Program()
            with paddle.static.program_guard(main_prog, startup_prog):
                weight_attr = paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Orthogonal(gain=gain)
                )
                linear = paddle.nn.Linear(6, 6, weight_attr=weight_attr)

            exe = paddle.static.Executor(paddle.CPUPlace())
            exe.run(startup_prog)
            result = exe.run(main_prog, fetch_list=[linear.weight])
            weight = result[0]
            # W * W^T / gain^2 应接近单位矩阵
            # W * W^T / gain^2 should be close to identity
            wwt = np.matmul(weight, weight.T) / (gain**2)
            identity = np.eye(6)
            np.testing.assert_allclose(wwt, identity, atol=1e-4)
        finally:
            paddle.disable_static()


if __name__ == '__main__':
    unittest.main()
