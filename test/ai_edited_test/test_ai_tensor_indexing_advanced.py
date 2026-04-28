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
张量索引高级测试 / Advanced Tensor Indexing Tests

测试目标 / Test Target:
  paddle.tensor 索引操作

覆盖的模块 / Covered Modules:
  - 布尔索引
  - 花式索引
  - 切片操作
  - paddle.index_put: 索引赋值
  - paddle.put_along_axis: 沿轴赋值

作用 / Purpose:
  补充张量索引API的测试，提升覆盖率。
"""

import unittest

import numpy as np

import paddle

paddle.disable_static()


class TestBasicIndexing(unittest.TestCase):
    """测试基本索引 / Test basic indexing"""

    def test_scalar_index(self):
        """测试标量索引 / Test scalar indexing"""
        x = paddle.to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = x[0, 1]
        self.assertAlmostEqual(float(result.numpy()), 2.0)

    def test_slice_indexing(self):
        """测试切片索引 / Test slice indexing"""
        x = paddle.arange(10).reshape([2, 5])
        result = x[:, 1:4]
        self.assertEqual(result.shape, [2, 3])

    def test_negative_index(self):
        """测试负索引 / Test negative indexing"""
        x = paddle.to_tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertAlmostEqual(float(x[-1].numpy()), 5.0)
        self.assertAlmostEqual(float(x[-2].numpy()), 4.0)

    def test_ellipsis(self):
        """测试省略号索引 / Test ellipsis indexing"""
        x = paddle.randn([4, 3, 8, 8])
        result = x[..., :4]
        self.assertEqual(result.shape, [4, 3, 8, 4])

    def test_step_slice(self):
        """测试步幅切片 / Test step slice"""
        x = paddle.arange(10)
        result = x[::2]
        np.testing.assert_allclose(result.numpy(), [0, 2, 4, 6, 8])


class TestBooleanIndexing(unittest.TestCase):
    """测试布尔索引 / Test boolean indexing"""

    def test_bool_mask_select(self):
        """测试布尔掩码选择 / Test boolean mask selection"""
        x = paddle.to_tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        mask = x > 2.5
        result = x[mask]
        np.testing.assert_allclose(result.numpy(), [3.0, 4.0, 5.0])

    def test_2d_bool_mask(self):
        """测试2D布尔掩码 / Test 2D boolean mask"""
        x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        mask = x > 3.0
        result = x[mask]
        np.testing.assert_allclose(result.numpy(), [4.0, 5.0, 6.0])


class TestFancyIndexing(unittest.TestCase):
    """测试花式索引 / Test fancy indexing"""

    def test_integer_array_index(self):
        """测试整数数组索引 / Test integer array indexing"""
        x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        indices = paddle.to_tensor([0, 2])
        result = x[indices]
        self.assertEqual(result.shape, [2, 2])
        np.testing.assert_allclose(result[0].numpy(), [1.0, 2.0])
        np.testing.assert_allclose(result[1].numpy(), [5.0, 6.0])

    def test_multi_dim_fancy_index(self):
        """测试多维花式索引 / Test multi-dimensional fancy indexing"""
        x = paddle.randn([5, 4, 3])
        row_idx = paddle.to_tensor([0, 2, 4])
        col_idx = paddle.to_tensor([1, 2, 0])
        result = x[row_idx, col_idx]
        self.assertEqual(result.shape, [3, 3])


class TestIndexPut(unittest.TestCase):
    """测试索引赋值 / Test index put"""

    def test_index_put_basic(self):
        """测试基本索引赋值 / Test basic index put"""
        x = paddle.zeros([5])
        indices = (paddle.to_tensor([1, 3]),)
        values = paddle.to_tensor([10.0, 20.0])
        result = paddle.index_put(x, indices, values)
        np.testing.assert_allclose(result.numpy(), [0.0, 10.0, 0.0, 20.0, 0.0])

    def test_index_put_2d(self):
        """测试2D索引赋值 / Test 2D index put"""
        x = paddle.zeros([3, 3])
        row_idx = paddle.to_tensor([0, 2])
        col_idx = paddle.to_tensor([1, 2])
        indices = (row_idx, col_idx)
        values = paddle.to_tensor([5.0, 7.0])
        result = paddle.index_put(x, indices, values)
        self.assertAlmostEqual(float(result[0, 1].numpy()), 5.0)
        self.assertAlmostEqual(float(result[2, 2].numpy()), 7.0)


class TestPutAlongAxis(unittest.TestCase):
    """测试沿轴赋值 / Test put along axis"""

    def test_put_along_axis(self):
        """测试沿轴赋值操作 / Test put_along_axis operation"""
        x = paddle.zeros([3, 4])
        indices = paddle.to_tensor([[1, 2, 3, 0], [0, 1, 2, 3], [2, 0, 1, 3]])
        values = paddle.ones([3, 4])
        result = paddle.put_along_axis(x, indices, values, axis=1)
        self.assertEqual(result.shape, [3, 4])

    def test_take_along_axis(self):
        """测试沿轴取值 / Test take_along_axis operation"""
        x = paddle.to_tensor([[4.0, 3.0, 5.0], [1.0, 2.0, 6.0]])
        indices = paddle.to_tensor([[2, 0], [2, 1]])
        result = paddle.take_along_axis(x, indices, axis=1)
        np.testing.assert_allclose(result.numpy(), [[5.0, 4.0], [6.0, 2.0]])


if __name__ == '__main__':
    unittest.main()
