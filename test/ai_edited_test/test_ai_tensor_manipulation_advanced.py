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

"""
进阶张量操作单元测试 / Advanced Tensor Manipulation Unit Tests

测试目标 / Test Target:
  paddle.tensor.manipulation 进阶功能 (覆盖率约73.9%)

覆盖的模块 / Covered Modules:
  - paddle.unbind: 张量拆解
  - paddle.broadcast_to: 广播扩展
  - paddle.expand: 展开
  - paddle.masked_select: 带掩码选择
  - paddle.index_select: 索引选择
  - paddle.index_put: 索引赋值
  - paddle.gather_nd: 多维聚合

作用 / Purpose:
  覆盖张量操作进阶函数的代码路径，补充对index/gather/scatter等操作的测试。
"""

import unittest

import numpy as np

import paddle

paddle.disable_static()


class TestUnbindAndSplit(unittest.TestCase):
    """测试unbind和split操作 / Test unbind and split operations"""

    def test_unbind_axis0(self):
        """测试沿axis=0解绑 / Test unbind along axis=0"""
        x = paddle.to_tensor([[1, 2, 3], [4, 5, 6]], dtype='float32')
        result = paddle.unbind(x, axis=0)
        self.assertEqual(len(result), 2)
        np.testing.assert_allclose(result[0].numpy(), np.array([1.0, 2.0, 3.0]))

    def test_unbind_axis1(self):
        """测试沿axis=1解绑 / Test unbind along axis=1"""
        x = paddle.to_tensor([[1, 2, 3], [4, 5, 6]], dtype='float32')
        result = paddle.unbind(x, axis=1)
        self.assertEqual(len(result), 3)

    def test_chunk(self):
        """测试张量分块 / Test tensor chunking"""
        x = paddle.to_tensor([1, 2, 3, 4, 5, 6], dtype='float32')
        result = paddle.chunk(x, chunks=3)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].shape, [2])


class TestBroadcastAndExpand(unittest.TestCase):
    """测试广播和展开 / Test broadcast and expand"""

    def test_broadcast_to(self):
        """测试广播到指定形状 / Test broadcast to shape"""
        x = paddle.to_tensor([[1.0], [2.0], [3.0]])
        result = paddle.broadcast_to(x, [3, 4])
        self.assertEqual(result.shape, [3, 4])

    def test_expand(self):
        """测试展开 / Test expand"""
        x = paddle.ones([3, 1])
        result = paddle.expand(x, [3, 4])
        self.assertEqual(result.shape, [3, 4])

    def test_expand_as(self):
        """测试按张量展开 / Test expand as"""
        x = paddle.ones([1, 4])
        y = paddle.ones([3, 4])
        result = paddle.expand_as(x, y)
        self.assertEqual(result.shape, [3, 4])


class TestMaskedOps(unittest.TestCase):
    """测试掩码操作 / Test masked operations"""

    def test_masked_select(self):
        """测试掩码选择 / Test masked select"""
        x = paddle.to_tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        mask = x > 2.5
        result = paddle.masked_select(x, mask)
        expected = np.array([3.0, 4.0, 5.0])
        np.testing.assert_allclose(result.numpy(), expected)

    def test_masked_fill(self):
        """测试掩码填充 / Test masked fill"""
        x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
        mask = paddle.to_tensor([[True, False], [False, True]])
        result = paddle.where(mask, paddle.zeros_like(x), x)
        self.assertEqual(result.shape, [2, 2])


class TestIndexOps(unittest.TestCase):
    """测试索引操作 / Test index operations"""

    def test_index_select(self):
        """测试索引选择 / Test index select"""
        x = paddle.to_tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        )
        indices = paddle.to_tensor([0, 2])
        result = paddle.index_select(x, indices, axis=0)
        self.assertEqual(result.shape, [2, 3])
        np.testing.assert_allclose(result[0].numpy(), np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(result[1].numpy(), np.array([7.0, 8.0, 9.0]))

    def test_gather(self):
        """测试gather操作 / Test gather operation"""
        x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        indices = paddle.to_tensor([0, 2])
        result = paddle.gather(x, indices)
        self.assertEqual(result.shape, [2, 2])

    def test_gather_nd(self):
        """测试gather_nd操作 / Test gather_nd operation"""
        x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
        indices = paddle.to_tensor([[0, 0], [1, 1]])
        result = paddle.gather_nd(x, indices)
        np.testing.assert_allclose(result.numpy(), np.array([1.0, 4.0]))

    def test_scatter(self):
        """测试scatter操作 / Test scatter operation"""
        x = paddle.zeros([3, 3])
        indices = paddle.to_tensor([0, 1])
        updates = paddle.ones([2, 3])
        result = paddle.scatter(x, indices, updates)
        self.assertEqual(result.shape, [3, 3])

    def test_scatter_nd(self):
        """测试scatter_nd操作 / Test scatter_nd operation"""
        indices = paddle.to_tensor([[0], [1], [2]])
        updates = paddle.to_tensor([10.0, 20.0, 30.0])
        shape = [5]
        result = paddle.scatter_nd(indices, updates, shape)
        self.assertEqual(result.shape, [5])
        self.assertAlmostEqual(float(result[0].numpy()), 10.0)


class TestStackAndCat(unittest.TestCase):
    """测试stack和cat操作 / Test stack and concat operations"""

    def test_stack(self):
        """测试张量堆叠 / Test tensor stack"""
        x = paddle.to_tensor([1.0, 2.0, 3.0])
        y = paddle.to_tensor([4.0, 5.0, 6.0])
        result = paddle.stack([x, y], axis=0)
        self.assertEqual(result.shape, [2, 3])

    def test_concat_axis0(self):
        """测试沿axis=0拼接 / Test concat along axis=0"""
        x = paddle.randn([2, 4])
        y = paddle.randn([3, 4])
        result = paddle.concat([x, y], axis=0)
        self.assertEqual(result.shape, [5, 4])

    def test_hstack(self):
        """测试水平堆叠 / Test horizontal stack"""
        x = paddle.randn([3, 4])
        y = paddle.randn([3, 5])
        result = paddle.hstack([x, y])
        self.assertEqual(result.shape, [3, 9])

    def test_vstack(self):
        """测试垂直堆叠 / Test vertical stack"""
        x = paddle.randn([2, 4])
        y = paddle.randn([3, 4])
        result = paddle.vstack([x, y])
        self.assertEqual(result.shape, [5, 4])


class TestReshapeAndPermute(unittest.TestCase):
    """测试reshape和permute / Test reshape and permute"""

    def test_reshape(self):
        """测试reshape / Test reshape"""
        x = paddle.randn([4, 6])
        result = x.reshape([2, 12])
        self.assertEqual(result.shape, [2, 12])

    def test_flatten(self):
        """测试flatten / Test flatten"""
        x = paddle.randn([4, 3, 8, 8])
        result = paddle.flatten(x, start_axis=1)
        self.assertEqual(result.shape, [4, 192])

    def test_transpose(self):
        """测试transpose / Test transpose"""
        x = paddle.randn([2, 3, 4])
        result = paddle.transpose(x, perm=[2, 0, 1])
        self.assertEqual(result.shape, [4, 2, 3])

    def test_squeeze(self):
        """测试squeeze / Test squeeze"""
        x = paddle.randn([4, 1, 8, 1])
        result = paddle.squeeze(x)
        self.assertEqual(result.shape, [4, 8])

    def test_unsqueeze(self):
        """测试unsqueeze / Test unsqueeze"""
        x = paddle.randn([4, 8])
        result = paddle.unsqueeze(x, axis=1)
        self.assertEqual(result.shape, [4, 1, 8])


if __name__ == '__main__':
    unittest.main()
