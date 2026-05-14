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

# [AUTO-GENERATED] test for paddle/tensor/logic.py
# Target file: python/paddle/tensor/logic.py
# Coverage: 75.7% (143/189) - Uncovered lines: 152-158,197-204,259,278,297-305,359,378,397-405,475,494,513-521,576,595,614-623,689,708,727-736,789,797
# 本文件为 tensor/logic.py 的单元测试 / Unit tests for tensor/logic.py
#
# 测试目标：
# - logical_and_, logical_or_, logical_xor_, logical_not_ 原地操作
# - is_empty 判断空张量
# - equal_all 判断全等
# - equal, greater_equal, less_equal, less_than, not_equal 比较操作
# - equal_, greater_equal_, less_equal_, less_than_, less_, not_equal_ 原地操作
# - is_tensor 判断张量
# - bitwise_invert / bitwise_invert_ 按位取反
# - __rand__, __ror__, __rxor__ 反向运算

import unittest

import numpy as np

import paddle


class TestLogicalInplaceOps(unittest.TestCase):
    """逻辑运算原地操作测试 / Logical inplace operation tests"""

    def setUp(self):
        """测试前准备 / Setup before tests"""
        paddle.disable_static()

    def tearDown(self):
        """测试后清理 / Teardown after tests"""
        pass

    def test_logical_and_inplace(self):
        """测试原地逻辑与操作 / Test inplace logical_and"""
        x = paddle.to_tensor([True, True, False, False], dtype='bool')
        y = paddle.to_tensor([True, False, True, False], dtype='bool')
        result = paddle.tensor.logic.logical_and_(x, y)
        expected = np.array([True, False, False, False])
        np.testing.assert_array_equal(result.numpy(), expected)
        # 验证是原地操作 / Verify inplace
        np.testing.assert_array_equal(x.numpy(), expected)

    def test_logical_or_inplace(self):
        """测试原地逻辑或操作 / Test inplace logical_or"""
        x = paddle.to_tensor([True, True, False, False], dtype='bool')
        y = paddle.to_tensor([True, False, True, False], dtype='bool')
        result = paddle.tensor.logic.logical_or_(x, y)
        expected = np.array([True, True, True, False])
        np.testing.assert_array_equal(result.numpy(), expected)
        np.testing.assert_array_equal(x.numpy(), expected)

    def test_logical_xor_inplace(self):
        """测试原地逻辑异或操作 / Test inplace logical_xor"""
        x = paddle.to_tensor([True, True, False, False], dtype='bool')
        y = paddle.to_tensor([True, False, True, False], dtype='bool')
        result = paddle.tensor.logic.logical_xor_(x, y)
        expected = np.array([False, True, True, False])
        np.testing.assert_array_equal(result.numpy(), expected)
        np.testing.assert_array_equal(x.numpy(), expected)

    def test_logical_not_inplace(self):
        """测试原地逻辑非操作 / Test inplace logical_not"""
        x = paddle.to_tensor([True, False, True, False], dtype='bool')
        result = paddle.tensor.logic.logical_not_(x)
        expected = np.array([False, True, False, True])
        np.testing.assert_array_equal(result.numpy(), expected)
        np.testing.assert_array_equal(x.numpy(), expected)

    def test_logical_and_inplace_broadcast_error(self):
        """测试原地逻辑与广播形状不匹配报错 / Test inplace logical_and broadcast shape mismatch"""
        x = paddle.to_tensor([[True, False]], dtype='bool')  # shape [1, 2]
        y = paddle.to_tensor([True, False, True], dtype='bool')  # shape [3]
        with self.assertRaises(ValueError):
            paddle.tensor.logic.logical_and_(x, y)

    def test_logical_or_inplace_broadcast_error(self):
        """测试原地逻辑或广播形状不匹配报错 / Test inplace logical_or broadcast shape mismatch"""
        x = paddle.to_tensor([[True, False]], dtype='bool')  # shape [1, 2]
        y = paddle.to_tensor([True, False, True], dtype='bool')  # shape [3]
        with self.assertRaises(ValueError):
            paddle.tensor.logic.logical_or_(x, y)

    def test_logical_xor_inplace_broadcast_error(self):
        """测试原地逻辑异或广播形状不匹配报错 / Test inplace logical_xor broadcast shape mismatch"""
        x = paddle.to_tensor([[True, False]], dtype='bool')
        y = paddle.to_tensor([True, False, True], dtype='bool')
        with self.assertRaises(ValueError):
            paddle.tensor.logic.logical_xor_(x, y)


class TestIsEmpty(unittest.TestCase):
    """is_empty 测试 / is_empty tests"""

    def setUp(self):
        """测试前准备 / Setup before tests"""
        paddle.disable_static()

    def tearDown(self):
        """测试后清理 / Teardown after tests"""
        pass

    def test_is_empty_non_empty_tensor(self):
        """测试非空张量返回 False / Test non-empty tensor returns False"""
        x = paddle.rand(shape=[4, 32, 32], dtype='float32')
        result = paddle.is_empty(x=x)
        self.assertFalse(result.item())

    def test_is_empty_different_dtypes(self):
        """测试不同数据类型的非空张量 / Test is_empty with different dtypes"""
        for dtype in ['float32', 'float64', 'int32', 'int64']:
            x = paddle.zeros(shape=[2, 3], dtype=dtype)
            result = paddle.is_empty(x=x)
            self.assertFalse(result.item())


class TestEqualAll(unittest.TestCase):
    """equal_all 测试 / equal_all tests"""

    def setUp(self):
        """测试前准备 / Setup before tests"""
        paddle.disable_static()

    def tearDown(self):
        """测试后清理 / Teardown after tests"""
        pass

    def test_equal_all_same(self):
        """测试相同张量返回 True / Test equal tensors return True"""
        x = paddle.to_tensor([1, 2, 3], dtype='float32')
        y = paddle.to_tensor([1, 2, 3], dtype='float32')
        result = paddle.equal_all(x, y)
        self.assertTrue(result.item())

    def test_equal_all_different(self):
        """测试不同张量返回 False / Test different tensors return False"""
        x = paddle.to_tensor([1, 2, 3], dtype='float32')
        y = paddle.to_tensor([1, 4, 3], dtype='float32')
        result = paddle.equal_all(x, y)
        self.assertFalse(result.item())

    def test_equal_all_same_object(self):
        """测试同一对象返回 True / Test same object returns True"""
        x = paddle.to_tensor([1, 2, 3], dtype='float32')
        result = paddle.equal_all(x, x)
        self.assertTrue(result.item())

    def test_equal_all_same_dtypes(self):
        """测试不同数据类型但值相同的张量 / Test same dtype tensors"""
        x = paddle.to_tensor([1, 2, 3], dtype='int64')
        y = paddle.to_tensor([1, 2, 3], dtype='int64')
        result = paddle.equal_all(x, y)
        self.assertTrue(result.item())

    def test_equal_all_different_values(self):
        """测试不同值但相同数据类型 / Test different values but same dtype"""
        x = paddle.to_tensor([1, 2, 3], dtype='int64')
        y = paddle.to_tensor([1, 4, 3], dtype='int64')
        result = paddle.equal_all(x, y)
        self.assertFalse(result.item())


class TestEqualComparison(unittest.TestCase):
    """equal 比较操作测试 / equal comparison tests"""

    def setUp(self):
        """测试前准备 / Setup before tests"""
        paddle.disable_static()

    def tearDown(self):
        """测试后清理 / Teardown after tests"""
        pass

    def test_equal_basic(self):
        """测试基本逐元素相等比较 / Test basic element-wise equal"""
        x = paddle.to_tensor([1, 2, 3])
        y = paddle.to_tensor([1, 3, 2])
        result = paddle.equal(x, y)
        expected = np.array([True, False, False])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_equal_with_scalar(self):
        """测试与标量比较 / Test equal with scalar"""
        x = paddle.to_tensor([1, 2, 3])
        result = paddle.equal(x, 2)
        expected = np.array([False, True, False])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_equal_with_bool(self):
        """测试与布尔值比较 / Test equal with bool"""
        x = paddle.to_tensor([True, False, True])
        result = paddle.equal(x, True)
        expected = np.array([True, False, True])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_equal_with_float(self):
        """测试与浮点数比较 / Test equal with float"""
        x = paddle.to_tensor([1.0, 2.0, 3.0])
        result = paddle.equal(x, 2.0)
        expected = np.array([False, True, False])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_equal_invalid_type_raises(self):
        """测试无效类型输入报错 / Test invalid type input raises TypeError"""
        x = paddle.to_tensor([1, 2, 3])
        with self.assertRaises(TypeError):
            paddle.equal(x, "invalid_type")

    def test_equal_inplace(self):
        """测试原地相等操作 / Test inplace equal"""
        x = paddle.to_tensor([1, 2, 3])
        y = paddle.to_tensor([1, 3, 2])
        result = paddle.tensor.logic.equal_(x, y)
        expected = np.array([True, False, False])
        np.testing.assert_array_equal(result.numpy(), expected)
        np.testing.assert_array_equal(x.numpy(), expected)

    def test_equal_inplace_broadcast_error(self):
        """测试原地相等广播形状不匹配报错 / Test inplace equal broadcast shape mismatch"""
        x = paddle.to_tensor([1, 2])
        y = paddle.to_tensor([1, 2, 3])
        with self.assertRaises(ValueError):
            paddle.tensor.logic.equal_(x, y)


class TestComparisonOps(unittest.TestCase):
    """比较操作测试 / Comparison operation tests"""

    def setUp(self):
        """测试前准备 / Setup before tests"""
        paddle.disable_static()

    def tearDown(self):
        """测试后清理 / Teardown after tests"""
        pass

    def test_greater_equal_basic(self):
        """测试大于等于 / Test greater_equal"""
        x = paddle.to_tensor([1, 2, 3])
        y = paddle.to_tensor([1, 3, 2])
        result = paddle.greater_equal(x, y)
        expected = np.array([True, False, True])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_less_equal_basic(self):
        """测试小于等于 / Test less_equal"""
        x = paddle.to_tensor([1, 2, 3])
        y = paddle.to_tensor([1, 3, 2])
        result = paddle.less_equal(x, y)
        expected = np.array([True, True, False])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_less_than_basic(self):
        """测试小于 / Test less_than"""
        x = paddle.to_tensor([1, 2, 3])
        y = paddle.to_tensor([1, 3, 2])
        result = paddle.less_than(x, y)
        expected = np.array([False, True, False])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_not_equal_basic(self):
        """测试不等于 / Test not_equal"""
        x = paddle.to_tensor([1, 2, 3])
        y = paddle.to_tensor([1, 3, 2])
        result = paddle.not_equal(x, y)
        expected = np.array([False, True, True])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_greater_equal_inplace(self):
        """测试原地大于等于 / Test inplace greater_equal"""
        x = paddle.to_tensor([1, 2, 3])
        y = paddle.to_tensor([1, 3, 2])
        result = paddle.tensor.logic.greater_equal_(x, y)
        expected = np.array([True, False, True])
        np.testing.assert_array_equal(result.numpy(), expected)
        np.testing.assert_array_equal(x.numpy(), expected)

    def test_greater_equal_inplace_broadcast_error(self):
        """测试原地大于等于广播形状不匹配报错 / Test inplace greater_equal broadcast shape mismatch"""
        x = paddle.to_tensor([[1, 2]], dtype='int32')
        y = paddle.to_tensor([1, 2, 3], dtype='int32')
        with self.assertRaises(ValueError):
            paddle.tensor.logic.greater_equal_(x, y)

    def test_less_equal_inplace(self):
        """测试原地小于等于 / Test inplace less_equal"""
        x = paddle.to_tensor([1, 2, 3])
        y = paddle.to_tensor([1, 3, 2])
        result = paddle.tensor.logic.less_equal_(x, y)
        expected = np.array([True, True, False])
        np.testing.assert_array_equal(result.numpy(), expected)
        np.testing.assert_array_equal(x.numpy(), expected)

    def test_less_equal_inplace_broadcast_error(self):
        """测试原地小于等于广播形状不匹配报错 / Test inplace less_equal broadcast shape mismatch"""
        x = paddle.to_tensor([[1, 2]], dtype='int32')
        y = paddle.to_tensor([1, 2, 3], dtype='int32')
        with self.assertRaises(ValueError):
            paddle.tensor.logic.less_equal_(x, y)

    def test_less_than_inplace(self):
        """测试原地小于 / Test inplace less_than"""
        x = paddle.to_tensor([1, 2, 3])
        y = paddle.to_tensor([1, 3, 2])
        result = paddle.tensor.logic.less_than_(x, y)
        expected = np.array([False, True, False])
        np.testing.assert_array_equal(result.numpy(), expected)
        np.testing.assert_array_equal(x.numpy(), expected)

    def test_less_than_inplace_broadcast_error(self):
        """测试原地小于广播形状不匹配报错 / Test inplace less_than broadcast shape mismatch"""
        x = paddle.to_tensor([[1, 2]], dtype='int32')
        y = paddle.to_tensor([1, 2, 3], dtype='int32')
        with self.assertRaises(ValueError):
            paddle.tensor.logic.less_than_(x, y)

    def test_less_inplace_alias(self):
        """测试 less_ 是 less_than_ 的别名 / Test less_ is alias of less_than_"""
        x = paddle.to_tensor([1, 2, 3])
        y = paddle.to_tensor([1, 3, 2])
        result = paddle.tensor.logic.less_(x, y)
        expected = np.array([False, True, False])
        np.testing.assert_array_equal(result.numpy(), expected)
        np.testing.assert_array_equal(x.numpy(), expected)

    def test_not_equal_inplace(self):
        """测试原地不等于 / Test inplace not_equal"""
        x = paddle.to_tensor([1, 2, 3])
        y = paddle.to_tensor([1, 3, 2])
        result = paddle.tensor.logic.not_equal_(x, y)
        expected = np.array([False, True, True])
        np.testing.assert_array_equal(result.numpy(), expected)
        np.testing.assert_array_equal(x.numpy(), expected)

    def test_not_equal_inplace_broadcast_error(self):
        """测试原地不等于广播形状不匹配报错 / Test inplace not_equal broadcast shape mismatch"""
        x = paddle.to_tensor([[1, 2]], dtype='int32')
        y = paddle.to_tensor([1, 2, 3], dtype='int32')
        with self.assertRaises(ValueError):
            paddle.tensor.logic.not_equal_(x, y)

    def test_greater_than_inplace(self):
        """测试原地大于 / Test inplace greater_than"""
        x = paddle.to_tensor([1, 2, 3])
        y = paddle.to_tensor([0, 1, 2])
        result = paddle.tensor.logic.greater_than_(x, y)
        expected = np.array([True, True, True])
        np.testing.assert_array_equal(result.numpy(), expected)
        np.testing.assert_array_equal(x.numpy(), expected)

    def test_greater_than_inplace_broadcast_error(self):
        """测试原地大于广播形状不匹配报错 / Test inplace greater_than broadcast shape mismatch"""
        x = paddle.to_tensor([[1, 2]], dtype='int32')
        y = paddle.to_tensor([1, 2, 3], dtype='int32')
        with self.assertRaises(ValueError):
            paddle.tensor.logic.greater_than_(x, y)


class TestIsTensor(unittest.TestCase):
    """is_tensor 测试 / is_tensor tests"""

    def setUp(self):
        """测试前准备 / Setup before tests"""
        paddle.disable_static()

    def tearDown(self):
        """测试后清理 / Teardown after tests"""
        pass

    def test_is_tensor_true(self):
        """测试张量返回 True / Test tensor returns True"""
        x = paddle.rand(shape=[2, 3])
        self.assertTrue(paddle.is_tensor(x))

    def test_is_tensor_false_list(self):
        """测试列表返回 False / Test list returns False"""
        self.assertFalse(paddle.is_tensor([1, 2, 3]))

    def test_is_tensor_false_int(self):
        """测试整数返回 False / Test int returns False"""
        self.assertFalse(paddle.is_tensor(42))

    def test_is_tensor_false_str(self):
        """测试字符串返回 False / Test string returns False"""
        self.assertFalse(paddle.is_tensor("hello"))

    def test_is_tensor_false_none(self):
        """测试 None 返回 False / Test None returns False"""
        self.assertFalse(paddle.is_tensor(None))

    def test_is_tensor_false_dict(self):
        """测试字典返回 False / Test dict returns False"""
        self.assertFalse(paddle.is_tensor({"a": 1}))

    def test_is_tensor_obj_alias(self):
        """测试 obj 参数别名 / Test obj parameter alias"""
        x = paddle.rand(shape=[2, 3])
        # obj 是 x 的别名 / obj is alias of x
        self.assertTrue(paddle.is_tensor(obj=x))


class TestBitwiseInvert(unittest.TestCase):
    """bitwise_invert 测试 / bitwise_invert tests"""

    def setUp(self):
        """测试前准备 / Setup before tests"""
        paddle.disable_static()

    def tearDown(self):
        """测试后清理 / Teardown after tests"""
        pass

    def test_bitwise_invert_int64(self):
        """测试 int64 按位取反 / Test int64 bitwise invert"""
        x = paddle.to_tensor([-5, -1, 1])
        result = paddle.bitwise_invert(x)
        expected = np.array([4, 0, -2])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_bitwise_invert_inplace(self):
        """测试原地按位取反 / Test inplace bitwise invert"""
        x = paddle.to_tensor([-5, -1, 1])
        result = paddle.tensor.logic.bitwise_invert_(x)
        expected = np.array([4, 0, -2])
        np.testing.assert_array_equal(result.numpy(), expected)
        np.testing.assert_array_equal(x.numpy(), expected)

    def test_bitwise_invert_bool(self):
        """测试 bool 按位取反 / Test bool bitwise invert"""
        x = paddle.to_tensor([True, False, True])
        result = paddle.bitwise_invert(x)
        expected = np.array([False, True, False])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_bitwise_invert_int32(self):
        """测试 int32 按位取反 / Test int32 bitwise invert"""
        x = paddle.to_tensor([0, 1, -1], dtype='int32')
        result = paddle.bitwise_invert(x)
        expected = np.array([-1, -2, 0], dtype='int32')
        np.testing.assert_array_equal(result.numpy(), expected)


class TestReverseOperators(unittest.TestCase):
    """反向运算符测试 / Reverse operator tests"""

    def setUp(self):
        """测试前准备 / Setup before tests"""
        paddle.disable_static()

    def tearDown(self):
        """测试后清理 / Teardown after tests"""
        pass

    def test_rand_int(self):
        """测试整数与张量的按位与反向操作 / Test int & tensor reverse bitwise_and"""
        x = paddle.to_tensor([5, 3, 7], dtype='int64')
        result = paddle.tensor.logic.__rand__(x, 3)
        # 5 & 3 = 1, 3 & 3 = 3, 7 & 3 = 3
        expected = np.array([1, 3, 3])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_rand_bool(self):
        """测试布尔与张量的按位与反向操作 / Test bool & tensor reverse bitwise_and"""
        x = paddle.to_tensor([True, False, True], dtype='bool')
        result = paddle.tensor.logic.__rand__(x, True)
        expected = np.array([True, False, True])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_rand_invalid_type_raises(self):
        """测试无效类型反向操作报错 / Test reverse op with invalid type raises TypeError"""
        x = paddle.to_tensor([1, 2, 3])
        with self.assertRaises(TypeError):
            paddle.tensor.logic.__rand__(x, "invalid")

    def test_ror_int(self):
        """测试整数与张量的按位或反向操作 / Test int | tensor reverse bitwise_or"""
        x = paddle.to_tensor([5, 3, 7], dtype='int64')
        result = paddle.tensor.logic.__ror__(x, 3)
        # 5 | 3 = 7, 3 | 3 = 3, 7 | 3 = 7
        expected = np.array([7, 3, 7])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_ror_invalid_type_raises(self):
        """测试无效类型按位或反向操作报错 / Test reverse or with invalid type raises TypeError"""
        x = paddle.to_tensor([1, 2, 3])
        with self.assertRaises(TypeError):
            paddle.tensor.logic.__ror__(x, "invalid")

    def test_rxor_int(self):
        """测试整数与张量的按位异或反向操作 / Test int ^ tensor reverse bitwise_xor"""
        x = paddle.to_tensor([5, 3, 7], dtype='int64')
        result = paddle.tensor.logic.__rxor__(x, 3)
        # 5 ^ 3 = 6, 3 ^ 3 = 0, 7 ^ 3 = 4
        expected = np.array([6, 0, 4])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_rxor_invalid_type_raises(self):
        """测试无效类型按位异或反向操作报错 / Test reverse xor with invalid type raises TypeError"""
        x = paddle.to_tensor([1, 2, 3])
        with self.assertRaises(TypeError):
            paddle.tensor.logic.__rxor__(x, "invalid")


class TestComparisonOpsWithDifferentDtypes(unittest.TestCase):
    """不同数据类型比较测试 / Comparison with different dtypes"""

    def setUp(self):
        """测试前准备 / Setup before tests"""
        paddle.disable_static()

    def tearDown(self):
        """测试后清理 / Teardown after tests"""
        pass

    def test_greater_equal_float64(self):
        """测试 float64 大于等于 / Test float64 greater_equal"""
        x = paddle.to_tensor([1.0, 2.0, 3.0], dtype='float64')
        y = paddle.to_tensor([1.0, 3.0, 2.0], dtype='float64')
        result = paddle.greater_equal(x, y)
        expected = np.array([True, False, True])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_less_equal_int64(self):
        """测试 int64 小于等于 / Test int64 less_equal"""
        x = paddle.to_tensor([1, 2, 3], dtype='int64')
        y = paddle.to_tensor([1, 3, 2], dtype='int64')
        result = paddle.less_equal(x, y)
        expected = np.array([True, True, False])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_less_than_int32(self):
        """测试 int32 小于 / Test int32 less_than"""
        x = paddle.to_tensor([1, 2, 3], dtype='int32')
        y = paddle.to_tensor([2, 2, 2], dtype='int32')
        result = paddle.less_than(x, y)
        expected = np.array([True, False, False])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_not_equal_bool(self):
        """测试 bool 不等于 / Test bool not_equal"""
        x = paddle.to_tensor([True, False, True], dtype='bool')
        y = paddle.to_tensor([True, True, False], dtype='bool')
        result = paddle.not_equal(x, y)
        expected = np.array([False, True, True])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_comparison_2d_tensors(self):
        """测试 2D 张量比较 / Test 2D tensor comparison"""
        x = paddle.to_tensor([[1, 2], [3, 4]])
        y = paddle.to_tensor([[1, 3], [2, 4]])
        result = paddle.equal(x, y)
        expected = np.array([[True, False], [False, True]])
        np.testing.assert_array_equal(result.numpy(), expected)


if __name__ == '__main__':
    unittest.main()
