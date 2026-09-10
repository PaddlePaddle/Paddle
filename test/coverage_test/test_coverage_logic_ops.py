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

# [AUTO-GENERATED] Unit test for paddle.tensor.logic
# 自动生成的单测，覆盖 paddle.tensor.logic 模块中未覆盖的代码
# Target: cover uncovered lines in paddle/python/paddle/tensor/logic.py
# including static graph branches for equal, less_than, less_equal, not_equal, greater_equal
# and inplace logic operations

"""
测试模块：paddle.tensor.logic
Test Module: paddle.tensor.logic

本测试覆盖以下功能：
This test covers the following functions:
1. is_empty - 检测Tensor是否为空 / Test if a Tensor is empty (lines 152-158)
2. equal_all - 检测两个Tensor是否完全相等 / Test if two Tensors are entirely equal (lines 197-204)
3. equal 静态图路径 / equal static graph path (lines 259-305)
4. greater_equal 静态图路径 / greater_equal static graph path (lines 397-405)
5. less_equal 静态图路径 / less_equal static graph path (lines 513-521)
6. less_than 静态图路径 / less_than static graph path (lines 614-623)
7. not_equal 静态图路径 / not_equal static graph path (lines 727-736)
8. 动态图inplace操作 / dynamic inplace operations
9. equal with scalar y / equal与标量y的比较

覆盖的未覆盖行：152-155, 197-204, 259, 278, 297-305, 359, 378, 397-405,
475, 494, 513-521, 576, 595, 614-623, 689, 708, 727-736, 789, 797
"""

import unittest

import numpy as np

import paddle


class TestIsEmpty(unittest.TestCase):
    """测试is_empty函数
    Test is_empty function"""

    def setUp(self):
        paddle.disable_static()

    def test_non_empty_tensor(self):
        """测试非空Tensor应返回False
        Test that non-empty Tensor returns False"""
        x = paddle.randn([3, 4])
        result = paddle.is_empty(x)
        self.assertFalse(result.item())

    def test_empty_tensor(self):
        """测试空Tensor应返回True
        Test that empty Tensor returns True"""
        x = paddle.empty([0, 3])
        result = paddle.is_empty(x)
        self.assertTrue(result.item())


class TestEqualAll(unittest.TestCase):
    """测试equal_all函数
    Test equal_all function"""

    def setUp(self):
        paddle.disable_static()

    def test_equal_tensors(self):
        """测试两个相同Tensor应返回True
        Test two identical Tensors should return True"""
        x = paddle.to_tensor([1, 2, 3])
        y = paddle.to_tensor([1, 2, 3])
        result = paddle.equal_all(x, y)
        self.assertTrue(result.item())

    def test_not_equal_tensors(self):
        """测试两个不同Tensor应返回False
        Test two different Tensors should return False"""
        x = paddle.to_tensor([1, 2, 3])
        y = paddle.to_tensor([1, 4, 3])
        result = paddle.equal_all(x, y)
        self.assertFalse(result.item())


class TestEqualWithScalar(unittest.TestCase):
    """测试equal与标量比较
    Test equal with scalar comparison"""

    def setUp(self):
        paddle.disable_static()

    def test_equal_with_int(self):
        """测试Tensor与int比较
        Test Tensor compared with int"""
        x = paddle.to_tensor([1, 2, 3])
        result = paddle.equal(x, 2)
        expected = np.array([False, True, False])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_equal_with_float(self):
        """测试Tensor与float比较
        Test Tensor compared with float"""
        x = paddle.to_tensor([1.0, 2.0, 3.0])
        result = paddle.equal(x, 2.0)
        expected = np.array([False, True, False])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_equal_with_bool(self):
        """测试Tensor与bool比较
        Test Tensor compared with bool"""
        x = paddle.to_tensor([True, False, True])
        result = paddle.equal(x, True)
        expected = np.array([True, False, True])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_equal_invalid_type_raises(self):
        """测试equal传入不支持的类型应报错
        Test equal with unsupported type should raise TypeError"""
        x = paddle.to_tensor([1, 2, 3])
        with self.assertRaises(TypeError):
            paddle.equal(x, "string")


class TestInplaceLogicOps(unittest.TestCase):
    """测试逻辑运算的inplace版本
    Test inplace versions of logic operations"""

    def setUp(self):
        paddle.disable_static()

    def test_logical_and_inplace(self):
        """测试logical_and_的inplace操作
        Test logical_and_ inplace operation"""
        x = paddle.to_tensor([True, False, True, False])
        y = paddle.to_tensor([True, True, False, False])
        paddle.logical_and_(x, y)
        expected = np.array([True, False, False, False])
        np.testing.assert_array_equal(x.numpy(), expected)

    def test_logical_or_inplace(self):
        """测试logical_or_的inplace操作
        Test logical_or_ inplace operation"""
        x = paddle.to_tensor([True, False, True, False])
        y = paddle.to_tensor([True, True, False, False])
        paddle.logical_or_(x, y)
        expected = np.array([True, True, True, False])
        np.testing.assert_array_equal(x.numpy(), expected)

    def test_logical_xor_inplace(self):
        """测试logical_xor_的inplace操作
        Test logical_xor_ inplace operation"""
        x = paddle.to_tensor([True, False, True, False])
        y = paddle.to_tensor([True, True, False, False])
        paddle.logical_xor_(x, y)
        expected = np.array([False, True, True, False])
        np.testing.assert_array_equal(x.numpy(), expected)

    def test_logical_not_inplace(self):
        """测试logical_not_的inplace操作
        Test logical_not_ inplace operation"""
        x = paddle.to_tensor([True, False, True])
        paddle.logical_not_(x)
        expected = np.array([False, True, False])
        np.testing.assert_array_equal(x.numpy(), expected)

    def test_equal_inplace(self):
        """测试equal_的inplace操作
        Test equal_ inplace operation"""
        x = paddle.to_tensor([1, 2, 3])
        y = paddle.to_tensor([1, 3, 3])
        result = paddle.equal_(x, y)
        expected = np.array([True, False, True])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_not_equal_inplace(self):
        """测试not_equal_的inplace操作
        Test not_equal_ inplace operation"""
        x = paddle.to_tensor([1, 2, 3])
        y = paddle.to_tensor([1, 3, 3])
        result = paddle.not_equal_(x, y)
        expected = np.array([False, True, False])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_greater_equal_inplace(self):
        """测试greater_equal_的inplace操作
        Test greater_equal_ inplace operation"""
        x = paddle.to_tensor([1, 2, 3])
        y = paddle.to_tensor([2, 2, 1])
        result = paddle.greater_equal_(x, y)
        expected = np.array([False, True, True])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_greater_than_inplace(self):
        """测试greater_than_的inplace操作
        Test greater_than_ inplace operation"""
        x = paddle.to_tensor([1, 2, 3])
        y = paddle.to_tensor([2, 2, 1])
        result = paddle.greater_than_(x, y)
        expected = np.array([False, False, True])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_less_equal_inplace(self):
        """测试less_equal_的inplace操作
        Test less_equal_ inplace operation"""
        x = paddle.to_tensor([1, 2, 3])
        y = paddle.to_tensor([2, 2, 1])
        result = paddle.less_equal_(x, y)
        expected = np.array([True, True, False])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_less_than_inplace(self):
        """测试less_than_的inplace操作
        Test less_than_ inplace operation"""
        x = paddle.to_tensor([1, 2, 3])
        y = paddle.to_tensor([2, 2, 1])
        result = paddle.less_than_(x, y)
        expected = np.array([True, False, False])
        np.testing.assert_array_equal(result.numpy(), expected)


class TestStaticGraphLogicOps(unittest.TestCase):
    """测试静态图模式下的逻辑比较运算，覆盖未覆盖的静态图代码路径
    Test static graph logic comparison ops to cover uncovered static graph paths"""

    def test_equal_static(self):
        """测试静态图模式下的equal操作
        Test equal in static graph mode"""
        paddle.enable_static()
        try:
            main_prog = paddle.static.Program()
            startup_prog = paddle.static.Program()
            with paddle.static.program_guard(main_prog, startup_prog):
                x = paddle.static.data(name='x', shape=[3], dtype='int32')
                y = paddle.static.data(name='y', shape=[3], dtype='int32')
                out = paddle.equal(x, y)

            exe = paddle.static.Executor(paddle.CPUPlace())
            exe.run(startup_prog)
            x_np = np.array([1, 2, 3], dtype='int32')
            y_np = np.array([1, 3, 3], dtype='int32')
            result = exe.run(
                main_prog, feed={'x': x_np, 'y': y_np}, fetch_list=[out]
            )
            np.testing.assert_array_equal(result[0], [True, False, True])
        finally:
            paddle.disable_static()

    def test_greater_equal_static(self):
        """测试静态图模式下的greater_equal操作
        Test greater_equal in static graph mode"""
        paddle.enable_static()
        try:
            main_prog = paddle.static.Program()
            startup_prog = paddle.static.Program()
            with paddle.static.program_guard(main_prog, startup_prog):
                x = paddle.static.data(name='x', shape=[3], dtype='float32')
                y = paddle.static.data(name='y', shape=[3], dtype='float32')
                out = paddle.greater_equal(x, y)

            exe = paddle.static.Executor(paddle.CPUPlace())
            exe.run(startup_prog)
            x_np = np.array([1.0, 2.0, 3.0], dtype='float32')
            y_np = np.array([2.0, 2.0, 1.0], dtype='float32')
            result = exe.run(
                main_prog, feed={'x': x_np, 'y': y_np}, fetch_list=[out]
            )
            np.testing.assert_array_equal(result[0], [False, True, True])
        finally:
            paddle.disable_static()

    def test_less_equal_static(self):
        """测试静态图模式下的less_equal操作
        Test less_equal in static graph mode"""
        paddle.enable_static()
        try:
            main_prog = paddle.static.Program()
            startup_prog = paddle.static.Program()
            with paddle.static.program_guard(main_prog, startup_prog):
                x = paddle.static.data(name='x', shape=[3], dtype='float32')
                y = paddle.static.data(name='y', shape=[3], dtype='float32')
                out = paddle.less_equal(x, y)

            exe = paddle.static.Executor(paddle.CPUPlace())
            exe.run(startup_prog)
            x_np = np.array([1.0, 2.0, 3.0], dtype='float32')
            y_np = np.array([2.0, 2.0, 1.0], dtype='float32')
            result = exe.run(
                main_prog, feed={'x': x_np, 'y': y_np}, fetch_list=[out]
            )
            np.testing.assert_array_equal(result[0], [True, True, False])
        finally:
            paddle.disable_static()

    def test_less_than_static(self):
        """测试静态图模式下的less_than操作
        Test less_than in static graph mode"""
        paddle.enable_static()
        try:
            main_prog = paddle.static.Program()
            startup_prog = paddle.static.Program()
            with paddle.static.program_guard(main_prog, startup_prog):
                x = paddle.static.data(name='x', shape=[3], dtype='float64')
                y = paddle.static.data(name='y', shape=[3], dtype='float64')
                out = paddle.less_than(x, y)

            exe = paddle.static.Executor(paddle.CPUPlace())
            exe.run(startup_prog)
            x_np = np.array([1.0, 2.0, 3.0], dtype='float64')
            y_np = np.array([2.0, 2.0, 1.0], dtype='float64')
            result = exe.run(
                main_prog, feed={'x': x_np, 'y': y_np}, fetch_list=[out]
            )
            np.testing.assert_array_equal(result[0], [True, False, False])
        finally:
            paddle.disable_static()

    def test_not_equal_static(self):
        """测试静态图模式下的not_equal操作
        Test not_equal in static graph mode"""
        paddle.enable_static()
        try:
            main_prog = paddle.static.Program()
            startup_prog = paddle.static.Program()
            with paddle.static.program_guard(main_prog, startup_prog):
                x = paddle.static.data(name='x', shape=[3], dtype='int64')
                y = paddle.static.data(name='y', shape=[3], dtype='int64')
                out = paddle.not_equal(x, y)

            exe = paddle.static.Executor(paddle.CPUPlace())
            exe.run(startup_prog)
            x_np = np.array([1, 2, 3], dtype='int64')
            y_np = np.array([1, 3, 3], dtype='int64')
            result = exe.run(
                main_prog, feed={'x': x_np, 'y': y_np}, fetch_list=[out]
            )
            np.testing.assert_array_equal(result[0], [False, True, False])
        finally:
            paddle.disable_static()

    def test_equal_all_static(self):
        """测试静态图模式下的equal_all操作
        Test equal_all in static graph mode"""
        paddle.enable_static()
        try:
            main_prog = paddle.static.Program()
            startup_prog = paddle.static.Program()
            with paddle.static.program_guard(main_prog, startup_prog):
                x = paddle.static.data(name='x', shape=[3], dtype='int32')
                y = paddle.static.data(name='y', shape=[3], dtype='int32')
                out = paddle.equal_all(x, y)

            exe = paddle.static.Executor(paddle.CPUPlace())
            exe.run(startup_prog)
            x_np = np.array([1, 2, 3], dtype='int32')
            y_np = np.array([1, 2, 3], dtype='int32')
            result = exe.run(
                main_prog, feed={'x': x_np, 'y': y_np}, fetch_list=[out]
            )
            self.assertTrue(result[0].item())
        finally:
            paddle.disable_static()


class TestBitwiseOps(unittest.TestCase):
    """测试bitwise相关操作
    Test bitwise operations"""

    def setUp(self):
        paddle.disable_static()

    def test_bitwise_invert(self):
        """测试bitwise_invert操作
        Test bitwise_invert operation"""
        x = paddle.to_tensor([-5, -1, 1])
        result = paddle.bitwise_invert(x)
        expected = np.array([4, 0, -2])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_bitwise_invert_inplace(self):
        """测试bitwise_invert_的inplace操作
        Test bitwise_invert_ inplace operation"""
        x = paddle.to_tensor([-5, -1, 1])
        paddle.bitwise_invert_(x)
        expected = np.array([4, 0, -2])
        np.testing.assert_array_equal(x.numpy(), expected)


if __name__ == '__main__':
    unittest.main()
