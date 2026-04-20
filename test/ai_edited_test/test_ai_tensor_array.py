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
张量数组操作单元测试 / Tensor Array Operations Unit Tests

测试目标 / Test Target:
  paddle.tensor.array 模块 (python/paddle/tensor/array.py, 覆盖率约69.5%)

覆盖的模块 / Covered Modules:
  - paddle.take: 从张量中取值
  - paddle.put_along_axis: 沿轴放置值
  - paddle.take_along_axis: 沿轴取值
  - paddle.roll: 循环移位

作用 / Purpose:
  覆盖张量数组操作的各种路径，补充对索引取值、沿轴操作等函数的测试。
"""

import unittest

import numpy as np

import paddle

paddle.disable_static()


class TestTakeOp(unittest.TestCase):
    """测试paddle.take操作 / Test paddle.take operation"""

    def test_take_basic(self):
        """测试基本take操作 / Test basic take operation"""
        x = paddle.to_tensor(np.arange(12, dtype='float32').reshape(3, 4))
        indices = paddle.to_tensor([0, 5, 11])
        result = paddle.take(x, indices)
        expected = np.array([0, 5, 11], dtype='float32')
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_take_1d(self):
        """测试1D张量的take / Test take on 1D tensor"""
        x = paddle.to_tensor(np.array([10, 20, 30, 40, 50], dtype='float32'))
        indices = paddle.to_tensor([2, 0, 4])
        result = paddle.take(x, indices)
        np.testing.assert_array_equal(result.numpy(), np.array([30, 10, 50]))

    def test_take_int_input(self):
        """测试整数输入的take / Test take with int input"""
        x = paddle.to_tensor([1, 2, 3, 4, 5])
        indices = paddle.to_tensor([0, 2, 4])
        result = paddle.take(x, indices)
        self.assertEqual(result.shape, [3])

    def test_take_mode_raise(self):
        """测试take的raise模式 / Test take raise mode"""
        x = paddle.to_tensor(np.arange(5, dtype='float32'))
        indices = paddle.to_tensor([0, 2])
        result = paddle.take(x, indices, mode='raise')
        self.assertEqual(result.shape, [2])

    def test_take_mode_wrap(self):
        """测试take的wrap模式 / Test take wrap mode"""
        x = paddle.to_tensor(np.arange(5, dtype='float32'))
        indices = paddle.to_tensor([0, 7])
        result = paddle.take(x, indices, mode='wrap')
        self.assertEqual(result.shape, [2])

    def test_take_mode_clip(self):
        """测试take的clip模式 / Test take clip mode"""
        x = paddle.to_tensor(np.arange(5, dtype='float32'))
        indices = paddle.to_tensor([0, 100])
        result = paddle.take(x, indices, mode='clip')
        self.assertEqual(result.shape, [2])
        self.assertAlmostEqual(result.numpy()[1], 4.0)


class TestPutAlongAxisOp(unittest.TestCase):
    """测试paddle.put_along_axis操作 / Test paddle.put_along_axis operation"""

    def test_put_along_axis_basic(self):
        """测试基本put_along_axis / Test basic put_along_axis"""
        x = paddle.zeros([3, 4], dtype='float32')
        indices = paddle.to_tensor([[0, 1, 2]], dtype='int64')
        values = paddle.to_tensor([[10.0, 20.0, 30.0]])
        result = paddle.put_along_axis(x, indices, values, axis=1)
        self.assertEqual(result.shape, [3, 4])

    def test_put_along_axis_reduce_assign(self):
        """测试assign reduce模式 / Test assign reduce mode"""
        x = paddle.ones([3, 4], dtype='float32')
        indices = paddle.zeros([3, 2], dtype='int64')
        values = paddle.ones([3, 2])
        result = paddle.put_along_axis(
            x, indices, values, axis=1, reduce='assign'
        )
        self.assertEqual(result.shape, [3, 4])

    def test_put_along_axis_axis0(self):
        """测试沿axis=0的put_along_axis / Test put_along_axis along axis=0"""
        x = paddle.zeros([4, 3], dtype='float32')
        indices = paddle.to_tensor([[0, 1, 2]], dtype='int64')
        values = paddle.ones([1, 3])
        result = paddle.put_along_axis(x, indices, values, axis=0)
        self.assertEqual(result.shape, [4, 3])


class TestTakeAlongAxisOp(unittest.TestCase):
    """测试paddle.take_along_axis操作 / Test paddle.take_along_axis operation"""

    def test_take_along_axis_basic(self):
        """测试基本take_along_axis / Test basic take_along_axis"""
        x = paddle.to_tensor([[1, 2, 3], [4, 5, 6]], dtype='float32')
        indices = paddle.to_tensor([[2, 0], [1, 2]], dtype='int64')
        result = paddle.take_along_axis(x, indices, axis=1)
        self.assertEqual(result.shape, [2, 2])

    def test_take_along_axis_0(self):
        """测试沿axis=0的take_along_axis / Test take_along_axis along axis=0"""
        x = paddle.to_tensor([[1, 2, 3], [4, 5, 6]], dtype='float32')
        indices = paddle.to_tensor([[0, 1, 0]], dtype='int64')
        result = paddle.take_along_axis(x, indices, axis=0)
        self.assertEqual(result.shape, [1, 3])

    def test_take_along_axis_3d(self):
        """测试3D张量的take_along_axis / Test take_along_axis on 3D tensor"""
        x = paddle.randn([2, 3, 4])
        indices = paddle.zeros([2, 1, 4], dtype='int64')
        result = paddle.take_along_axis(x, indices, axis=1)
        self.assertEqual(result.shape, [2, 1, 4])


class TestRollOp(unittest.TestCase):
    """测试paddle.roll操作 / Test paddle.roll operation"""

    def test_roll_basic(self):
        """测试基本roll / Test basic roll"""
        x = paddle.to_tensor([1, 2, 3, 4, 5], dtype='float32')
        result = paddle.roll(x, shifts=2)
        expected = np.array([4, 5, 1, 2, 3], dtype='float32')
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_roll_negative_shift(self):
        """测试负移位的roll / Test roll with negative shift"""
        x = paddle.to_tensor([1, 2, 3, 4, 5], dtype='float32')
        result = paddle.roll(x, shifts=-1)
        expected = np.array([2, 3, 4, 5, 1], dtype='float32')
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_roll_2d_axis(self):
        """测试2D张量沿轴的roll / Test roll on 2D tensor along axis"""
        x = paddle.to_tensor([[1, 2, 3], [4, 5, 6]], dtype='float32')
        result = paddle.roll(x, shifts=1, axis=0)
        expected = np.array([[4, 5, 6], [1, 2, 3]], dtype='float32')
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_roll_multiple_shifts(self):
        """测试多轴roll / Test roll with multiple shifts"""
        x = paddle.to_tensor([[1, 2, 3], [4, 5, 6]], dtype='float32')
        result = paddle.roll(x, shifts=[1, 2], axis=[0, 1])
        self.assertEqual(result.shape, [2, 3])

    def test_roll_flatten(self):
        """测试展平的roll / Test roll on flattened tensor"""
        x = paddle.randn([3, 4])
        result = paddle.roll(x, shifts=5)
        self.assertEqual(result.shape, [3, 4])


class TestNanToNumOp(unittest.TestCase):
    """测试paddle.nan_to_num操作 / Test paddle.nan_to_num operation"""

    def test_nan_to_num_basic(self):
        """测试基本nan_to_num / Test basic nan_to_num"""
        x = paddle.to_tensor([float('nan'), 1.0, float('inf'), float('-inf')])
        result = paddle.nan_to_num(x)
        self.assertFalse(paddle.any(paddle.isnan(result)).item())

    def test_nan_to_num_with_values(self):
        """测试带替换值的nan_to_num / Test nan_to_num with replacement values"""
        x = paddle.to_tensor([float('nan'), 1.0, float('inf'), float('-inf')])
        result = paddle.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        self.assertAlmostEqual(result.numpy()[0], 0.0)


if __name__ == '__main__':
    unittest.main()
