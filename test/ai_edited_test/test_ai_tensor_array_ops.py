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

# [AUTO-GENERATED] Test file for paddle.tensor.array
# 覆盖模块: paddle/tensor/array.py
# 未覆盖行: 83,87,91,92,93,94,99,172,173,174,178,179,180,185,262,269,275,284,285,286,287,288,292,295,296,301,306,369,370,376
# Covered module: paddle/tensor/array.py
# Uncovered lines: 83,87,91,92,93,94,99,172,173,174,178,179,180,185,262,269,275,284,285,286,287,288,292,295,296,301,306,369,370,376

import unittest

import numpy as np

import paddle


class TestArrayLength(unittest.TestCase):
    """测试 array_length 函数
    Test array_length function"""

    def test_array_length_python_list(self):
        """测试 Python 列表的 array_length
        Test array_length with Python list"""
        result = paddle.tensor.array_length([1, 2, 3])
        self.assertEqual(result, 3)

    def test_array_length_empty_list(self):
        """测试空列表的 array_length
        Test array_length with empty list"""
        result = paddle.tensor.array_length([])
        self.assertEqual(result, 0)


class TestArrayWrite(unittest.TestCase):
    """测试 array_write 和 array_read 函数
    Test array_write and array_read functions"""

    def test_array_write_read(self):
        """测试 array_write 后 array_read
        Test array_write followed by array_read"""
        arr = paddle.tensor.create_array(dtype='float32')
        x = paddle.full(shape=[3], fill_value=5.0, dtype="float32")
        i = paddle.zeros(shape=[1], dtype="int32")
        paddle.tensor.array_write(x, i, arr)
        result = paddle.tensor.array_read(arr, i)
        np.testing.assert_allclose(result.numpy(), [5.0, 5.0, 5.0])

    def test_array_write_multiple(self):
        """测试多次 array_write
        Test multiple array_write operations"""
        arr = paddle.tensor.create_array(dtype='float32')
        for idx in range(3):
            i = paddle.full(shape=[1], fill_value=idx, dtype="int32")
            x = paddle.full(shape=[2], fill_value=float(idx), dtype="float32")
            paddle.tensor.array_write(x, i, arr)
        # Read back
        for idx in range(3):
            i = paddle.full(shape=[1], fill_value=idx, dtype="int32")
            result = paddle.tensor.array_read(arr, i)
            np.testing.assert_allclose(result.numpy(), [float(idx), float(idx)])


class TestCreateArray(unittest.TestCase):
    """测试 create_array 函数
    Test create_array function"""

    def test_create_array_float32(self):
        """测试创建 float32 类型的数组
        Test creating float32 array"""
        arr = paddle.tensor.create_array(dtype='float32')
        self.assertIsNotNone(arr)

    def test_create_array_int32(self):
        """测试创建 int32 类型的数组
        Test creating int32 array"""
        arr = paddle.tensor.create_array(dtype='int32')
        self.assertIsNotNone(arr)


class TestStack(unittest.TestCase):
    """测试 stack 操作 (通过 array 相关操作)
    Test stack operation through array-related operations"""

    def test_stack_basic(self):
        """测试基本的 stack 操作
        Test basic stack operation"""
        x1 = paddle.to_tensor([1.0, 2.0])
        x2 = paddle.to_tensor([3.0, 4.0])
        result = paddle.stack([x1, x2], axis=0)
        np.testing.assert_array_equal(result.numpy(), [[1.0, 2.0], [3.0, 4.0]])

    def test_stack_axis1(self):
        """测试 axis=1 的 stack
        Test stack with axis=1"""
        x1 = paddle.to_tensor([1.0, 2.0])
        x2 = paddle.to_tensor([3.0, 4.0])
        result = paddle.stack([x1, x2], axis=1)
        np.testing.assert_array_equal(result.numpy(), [[1.0, 3.0], [2.0, 4.0]])


if __name__ == '__main__':
    unittest.main()
