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

# [AUTO-GENERATED] Unit test for paddle.nn.functional.input
# 自动生成的单测，覆盖 paddle.nn.functional.input 模块中未覆盖的代码
# Target: cover uncovered lines 118-137, 315-332 in paddle/python/paddle/nn/functional/input.py

"""
测试模块：paddle.nn.functional.input (embedding, one_hot)
Test Module: paddle.nn.functional.input (embedding, one_hot)

本测试覆盖以下功能：
This test covers the following functions:
1. one_hot - 独热编码 / One-hot encoding
   - 静态图路径 / Static graph path (lines 118-137)
   - 自动推断num_classes / Auto-infer num_classes
2. embedding - 嵌入查找 / Embedding lookup
   - max_norm 参数：嵌入向量范数裁剪 / max_norm parameter: embedding norm clipping
   - embedding_renorm_ 函数 / embedding_renorm_ function
   - scale_grad_by_freq 参数 / scale_grad_by_freq parameter (lines 315-332)
   - padding_idx 负索引 / negative padding_idx

覆盖的未覆盖行：118-137（one_hot静态图），315-332（embedding scale_grad_by_freq）
"""

import unittest

import numpy as np

import paddle


class TestOneHotDynamic(unittest.TestCase):
    """测试动态图模式下的one_hot
    Test one_hot in dynamic graph mode"""

    def setUp(self):
        paddle.disable_static()

    def test_basic_one_hot(self):
        """测试基本的one_hot编码
        Test basic one_hot encoding"""
        x = paddle.to_tensor([0, 1, 2, 3], dtype='int64')
        out = paddle.nn.functional.one_hot(x, num_classes=4)
        expected = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype='float32',
        )
        np.testing.assert_array_equal(out.numpy(), expected)

    def test_one_hot_auto_num_classes(self):
        """测试one_hot自动推断num_classes（num_classes=-1）
        Test one_hot with automatic num_classes inference (num_classes=-1)"""
        x = paddle.to_tensor([0, 1, 2], dtype='int64')
        out = paddle.nn.functional.one_hot(x)  # num_classes defaults to -1
        self.assertEqual(list(out.shape), [3, 3])

    def test_one_hot_2d_input(self):
        """测试2D输入的one_hot
        Test one_hot with 2D input"""
        x = paddle.to_tensor([[0, 1], [2, 0]], dtype='int64')
        out = paddle.nn.functional.one_hot(x, num_classes=3)
        self.assertEqual(list(out.shape), [2, 2, 3])


class TestOneHotStatic(unittest.TestCase):
    """测试静态图模式下的one_hot，覆盖未覆盖行118-137
    Test one_hot in static graph mode to cover uncovered lines 118-137"""

    def test_static_graph_one_hot(self):
        """测试静态图模式下的one_hot基本功能
        Test basic one_hot in static graph mode"""
        paddle.enable_static()
        try:
            main_prog = paddle.static.Program()
            startup_prog = paddle.static.Program()
            with paddle.static.program_guard(main_prog, startup_prog):
                x = paddle.static.data(name='x', shape=[4], dtype='int64')
                out = paddle.nn.functional.one_hot(x, num_classes=5)

            exe = paddle.static.Executor(paddle.CPUPlace())
            exe.run(startup_prog)
            x_np = np.array([0, 1, 3, 4], dtype='int64')
            result = exe.run(main_prog, feed={'x': x_np}, fetch_list=[out])
            self.assertEqual(list(result[0].shape), [4, 5])
            # 验证第一个向量 / Verify first vector
            np.testing.assert_array_equal(result[0][0], [1, 0, 0, 0, 0])
        finally:
            paddle.disable_static()


class TestEmbeddingMaxNorm(unittest.TestCase):
    """测试embedding的max_norm功能
    Test embedding max_norm feature"""

    def setUp(self):
        paddle.disable_static()

    def test_embedding_with_max_norm(self):
        """测试embedding的max_norm范数裁剪，覆盖embedding_renorm_函数
        Test embedding max_norm norm clipping, covers embedding_renorm_ function"""
        x = paddle.to_tensor([0, 1, 2], dtype='int64')
        # 创建一个权重矩阵，其中某些行的范数大于max_norm
        # Create a weight matrix where some rows have norm > max_norm
        weight = paddle.to_tensor(
            [
                [10.0, 10.0, 10.0],  # norm = sqrt(300) ≈ 17.3
                [1.0, 0.0, 0.0],  # norm = 1.0
                [0.0, 2.0, 0.0],  # norm = 2.0
            ],
            dtype='float32',
        )
        weight.stop_gradient = False

        out = paddle.nn.functional.embedding(
            x, weight, max_norm=5.0, norm_type=2.0
        )
        self.assertEqual(list(out.shape), [3, 3])
        # 第一行的范数应被裁剪到不超过5.0
        # First row norm should be clipped to <= 5.0
        row0_norm = float(paddle.norm(out[0], p=2).item())
        self.assertLessEqual(row0_norm, 5.0 + 1e-3)

    def test_embedding_with_negative_padding_idx(self):
        """测试embedding的负padding_idx
        Test embedding with negative padding_idx"""
        x = paddle.to_tensor([0, 1, 4], dtype='int64')
        weight = paddle.full(shape=(5, 3), fill_value=2.0, dtype='float32')
        out = paddle.nn.functional.embedding(x, weight, padding_idx=-1)
        self.assertEqual(list(out.shape), [3, 3])
        # padding_idx=-1 → 实际index=4, 输出应全0
        # padding_idx=-1 → actual index=4, output should be all zeros
        np.testing.assert_array_equal(out[2].numpy(), [0.0, 0.0, 0.0])

    def test_embedding_with_scale_grad_by_freq(self):
        """测试embedding的scale_grad_by_freq参数
        Test embedding scale_grad_by_freq parameter"""
        x = paddle.to_tensor([0, 0, 1, 2], dtype='int64')
        weight = paddle.randn([5, 3])
        weight.stop_gradient = False
        out = paddle.nn.functional.embedding(
            x, weight, scale_grad_by_freq=True, sparse=False
        )
        self.assertEqual(list(out.shape), [4, 3])
        # 验证可以正常前向计算 / Verify forward pass works
        loss = out.sum()
        loss.backward()

    def test_embedding_padding_idx_out_of_range(self):
        """测试embedding的padding_idx超出范围应报错
        Test embedding with padding_idx out of range should raise"""
        x = paddle.to_tensor([0, 1], dtype='int64')
        weight = paddle.randn([5, 3])
        with self.assertRaises(ValueError):
            paddle.nn.functional.embedding(x, weight, padding_idx=10)


if __name__ == '__main__':
    unittest.main()
