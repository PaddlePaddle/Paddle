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

# [AUTO-GENERATED] test for paddle/tensor/array.py
# Target file: python/paddle/tensor/array.py
# Coverage: 69.5% (73/105) - Uncovered lines: 83,87,91-99,172-185,262,269,275,284-306,369-379
# 本文件为 tensor/array.py 的单元测试 / Unit tests for tensor/array.py
#
# 测试目标：
# - create_array() 创建数组（含 initialized_list 参数）
# - array_length() 获取数组长度
# - array_read() 读取数组元素
# - array_write() 写入数组元素
# - 动态模式下各种数组操作

import unittest

import numpy as np

import paddle


class TestCreateArrayDygraph(unittest.TestCase):
    """动态模式下 create_array 测试 / create_array tests in dygraph mode"""

    def setUp(self):
        """测试前准备 / Setup before tests"""
        paddle.disable_static()

    def tearDown(self):
        """测试后清理 / Teardown after tests"""
        pass

    def test_create_array_basic(self):
        """测试基本数组创建 / Test basic array creation"""
        arr = paddle.tensor.create_array(dtype='float32')
        self.assertIsInstance(arr, list)
        self.assertEqual(len(arr), 0)

    def test_create_array_with_initialized_list(self):
        """测试使用 initialized_list 创建数组 / Test create_array with initialized_list"""
        x1 = paddle.to_tensor([1.0, 2.0, 3.0], dtype='float32')
        x2 = paddle.to_tensor([4.0, 5.0, 6.0], dtype='float32')
        arr = paddle.tensor.create_array(
            dtype='float32', initialized_list=[x1, x2]
        )
        self.assertIsInstance(arr, list)
        self.assertEqual(len(arr), 2)

    def test_create_array_with_initialized_tuple(self):
        """测试使用元组初始化 / Test create_array with tuple initialization"""
        x1 = paddle.to_tensor([1.0], dtype='float32')
        x2 = paddle.to_tensor([2.0], dtype='float32')
        arr = paddle.tensor.create_array(
            dtype='float32', initialized_list=(x1, x2)
        )
        self.assertIsInstance(arr, list)
        self.assertEqual(len(arr), 2)

    def test_create_array_invalid_initialized_list(self):
        """测试使用非列表/元组初始化报错 / Test create_array with invalid initialized_list"""
        with self.assertRaises(TypeError):
            paddle.tensor.create_array(
                dtype='float32', initialized_list="not_a_list"
            )

    def test_create_array_non_tensor_in_list(self):
        """测试初始化列表包含非张量 / Test create_array with non-tensor in list"""
        with self.assertRaises(TypeError):
            paddle.tensor.create_array(
                dtype='float32', initialized_list=[1, 2, 3]
            )

    def test_create_array_different_dtypes(self):
        """测试不同数据类型创建数组 / Test create_array with different dtypes"""
        for dtype in ['float32', 'float64', 'int32', 'int64']:
            arr = paddle.tensor.create_array(dtype=dtype)
            self.assertIsInstance(arr, list)

    def test_create_array_empty_initialized_list(self):
        """测试空初始化列表 / Test create_array with empty initialized_list"""
        arr = paddle.tensor.create_array(dtype='float32', initialized_list=[])
        self.assertIsInstance(arr, list)
        self.assertEqual(len(arr), 0)


class TestArrayLengthDygraph(unittest.TestCase):
    """动态模式下 array_length 测试 / array_length tests in dygraph mode"""

    def setUp(self):
        """测试前准备 / Setup before tests"""
        paddle.disable_static()

    def tearDown(self):
        """测试后清理 / Teardown after tests"""
        pass

    def test_array_length_empty(self):
        """测试空数组长度 / Test empty array length"""
        arr = []
        length = paddle.tensor.array_length(arr)
        self.assertEqual(length, 0)

    def test_array_length_non_empty(self):
        """测试非空数组长度 / Test non-empty array length"""
        x1 = paddle.to_tensor([1.0], dtype='float32')
        x2 = paddle.to_tensor([2.0], dtype='float32')
        arr = [x1, x2]
        length = paddle.tensor.array_length(arr)
        self.assertEqual(length, 2)

    def test_array_length_not_list_raises(self):
        """测试非列表输入报错 / Test non-list input raises AssertionError"""
        with self.assertRaises(AssertionError):
            paddle.tensor.array_length("not_a_list")


class TestArrayWriteDygraph(unittest.TestCase):
    """动态模式下 array_write 测试 / array_write tests in dygraph mode"""

    def setUp(self):
        """测试前准备 / Setup before tests"""
        paddle.disable_static()

    def tearDown(self):
        """测试后清理 / Teardown after tests"""
        pass

    def test_array_write_basic(self):
        """测试基本写入操作 / Test basic write operation"""
        x = paddle.full(shape=[2, 3], fill_value=5, dtype="float32")
        i = paddle.zeros(shape=[1], dtype="int32")
        arr = paddle.tensor.create_array(dtype="float32")
        arr = paddle.tensor.array_write(x, i, array=arr)
        self.assertIsInstance(arr, list)
        self.assertEqual(len(arr), 1)

    def test_array_write_creates_new_array(self):
        """测试不传 array 参数自动创建新数组 / Test auto-creation when array is None"""
        x = paddle.full(shape=[2, 3], fill_value=5, dtype="float32")
        i = paddle.zeros(shape=[1], dtype="int32")
        arr = paddle.tensor.array_write(x, i)
        self.assertIsInstance(arr, list)
        self.assertEqual(len(arr), 1)

    def test_array_write_multiple(self):
        """测试多次写入 / Test multiple writes"""
        arr = paddle.tensor.create_array(dtype="float32")
        for k in range(5):
            x = paddle.full(shape=[1, 2], fill_value=k, dtype="float32")
            i = paddle.to_tensor([k], dtype='int32')
            arr = paddle.tensor.array_write(x, i, array=arr)
        self.assertEqual(len(arr), 5)

    def test_array_write_overwrite(self):
        """测试覆盖写入 / Test overwrite write"""
        arr = paddle.tensor.create_array(dtype="float32")
        x1 = paddle.full(shape=[1, 2], fill_value=1, dtype="float32")
        x2 = paddle.full(shape=[1, 2], fill_value=2, dtype="float32")
        i0 = paddle.zeros(shape=[1], dtype="int32")
        arr = paddle.tensor.array_write(x1, i0, array=arr)
        # 写入到相同索引，应覆盖 / Write to same index, should overwrite
        arr = paddle.tensor.array_write(x2, i0, array=arr)
        self.assertEqual(len(arr), 1)
        # 验证覆盖后的值 / Verify overwritten value
        result = arr[0].numpy()
        np.testing.assert_array_equal(
            result, np.full((1, 2), 2, dtype='float32')
        )

    def test_array_write_append(self):
        """测试追加写入 / Test append write"""
        arr = paddle.tensor.create_array(dtype="float32")
        x0 = paddle.full(shape=[1, 2], fill_value=0, dtype="float32")
        x1 = paddle.full(shape=[1, 2], fill_value=1, dtype="float32")
        x2 = paddle.full(shape=[1, 2], fill_value=2, dtype="float32")
        i0 = paddle.zeros(shape=[1], dtype="int32")
        i1 = paddle.to_tensor([1], dtype='int32')
        i2 = paddle.to_tensor([2], dtype='int32')
        arr = paddle.tensor.array_write(x0, i0, array=arr)
        arr = paddle.tensor.array_write(x1, i1, array=arr)
        arr = paddle.tensor.array_write(x2, i2, array=arr)
        self.assertEqual(len(arr), 3)


class TestArrayReadDygraph(unittest.TestCase):
    """动态模式下 array_read 测试 / array_read tests in dygraph mode"""

    def setUp(self):
        """测试前准备 / Setup before tests"""
        paddle.disable_static()

    def tearDown(self):
        """测试后清理 / Teardown after tests"""
        pass

    def test_array_read_basic(self):
        """测试基本读取操作 / Test basic read operation"""
        arr = paddle.tensor.create_array(dtype="float32")
        x = paddle.full(shape=[1, 3], fill_value=5, dtype="float32")
        i = paddle.zeros(shape=[1], dtype="int32")
        arr = paddle.tensor.array_write(x, i, array=arr)
        item = paddle.tensor.array_read(arr, i)
        np.testing.assert_array_equal(
            item.numpy(), np.full((1, 3), 5, dtype='float32')
        )

    def test_array_read_multiple_indices(self):
        """测试读取多个位置 / Test reading from multiple positions"""
        arr = paddle.tensor.create_array(dtype="float32")
        values = []
        for k in range(3):
            x = paddle.full(shape=[1, 2], fill_value=k + 10, dtype="float32")
            i = paddle.to_tensor([k], dtype='int32')
            arr = paddle.tensor.array_write(x, i, array=arr)
            values.append((k + 10) * np.ones((1, 2), dtype='float32'))

        for k in range(3):
            i = paddle.to_tensor([k], dtype='int32')
            item = paddle.tensor.array_read(arr, i)
            np.testing.assert_array_equal(item.numpy(), values[k])

    def test_array_read_not_list_raises(self):
        """测试非列表输入报错 / Test non-list input raises AssertionError"""
        i = paddle.zeros(shape=[1], dtype="int32")
        with self.assertRaises(AssertionError):
            paddle.tensor.array_read("not_a_list", i)

    def test_array_read_invalid_index_shape_raises(self):
        """测试索引形状不对报错 / Test wrong index shape raises AssertionError"""
        arr = [paddle.to_tensor([1.0])]
        i = paddle.zeros(shape=[2], dtype="int32")  # shape should be [1]
        with self.assertRaises(AssertionError):
            paddle.tensor.array_read(arr, i)


class TestArrayIntegrationDygraph(unittest.TestCase):
    """动态模式下数组操作集成测试 / Array operation integration tests in dygraph mode"""

    def setUp(self):
        """测试前准备 / Setup before tests"""
        paddle.disable_static()

    def tearDown(self):
        """测试后清理 / Teardown after tests"""
        pass

    def test_full_write_read_cycle(self):
        """测试完整的写入-读取循环 / Test full write-read cycle"""
        arr = paddle.tensor.create_array(dtype="float32")
        # 写入多个元素 / Write multiple elements
        for k in range(4):
            x = paddle.full(shape=[2, 2], fill_value=k, dtype="float32")
            i = paddle.to_tensor([k], dtype='int32')
            arr = paddle.tensor.array_write(x, i, array=arr)

        # 验证长度 / Verify length
        self.assertEqual(paddle.tensor.array_length(arr), 4)

        # 读取并验证所有元素 / Read and verify all elements
        for k in range(4):
            i = paddle.to_tensor([k], dtype='int32')
            item = paddle.tensor.array_read(arr, i)
            np.testing.assert_array_equal(
                item.numpy(), np.full((2, 2), k, dtype='float32')
            )

    def test_initialized_list_write_read(self):
        """测试使用初始化列表进行写入读取 / Test write-read with initialized list"""
        x1 = paddle.full(shape=[1, 3], fill_value=100, dtype="float32")
        x2 = paddle.full(shape=[1, 3], fill_value=200, dtype="float32")
        arr = paddle.tensor.create_array(
            dtype='float32', initialized_list=[x1, x2]
        )

        self.assertEqual(paddle.tensor.array_length(arr), 2)

        i0 = paddle.zeros(shape=[1], dtype="int32")
        item0 = paddle.tensor.array_read(arr, i0)
        np.testing.assert_array_equal(
            item0.numpy(), np.full((1, 3), 100, dtype='float32')
        )

        i1 = paddle.to_tensor([1], dtype='int32')
        item1 = paddle.tensor.array_read(arr, i1)
        np.testing.assert_array_equal(
            item1.numpy(), np.full((1, 3), 200, dtype='float32')
        )

    def test_int64_array_operations(self):
        """测试 int64 类型数组操作 / Test int64 array operations"""
        arr = paddle.tensor.create_array(dtype='int64')
        x = paddle.full(shape=[2, 2], fill_value=42, dtype="int64")
        i = paddle.zeros(shape=[1], dtype="int32")
        arr = paddle.tensor.array_write(x, i, array=arr)
        item = paddle.tensor.array_read(arr, i)
        np.testing.assert_array_equal(
            item.numpy(), np.full((2, 2), 42, dtype='int64')
        )


if __name__ == '__main__':
    unittest.main()
