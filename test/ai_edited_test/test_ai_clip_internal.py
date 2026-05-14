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

# [AUTO-GENERATED] Unit test for paddle.nn.clip
# Target: cover uncovered lines 222-234, 237-248, 251-260, 263-283, 286-289, 294-352
# in paddle/nn/clip.py

"""
测试模块：paddle.nn.clip - 内部函数和 ErrorClipByValue
Test Module: paddle.nn.clip - internal functions and ErrorClipByValue

本测试覆盖以下功能：
This test covers:
1. _clip_by_global_norm_using_mp_type - 全局混合精度裁剪标志控制
2. _cast_to_mp_type_if_enabled - 混合精度类型转换
3. _can_inplace_clip_grad - 判断梯度是否可原地裁剪
4. _squared_l2_norm - 平方 L2 范数计算（动态图）
5. ErrorClipByValue - 按值错误裁剪类的构造和字符串表示
6. BaseErrorClipAttr - 基础错误裁剪类接口
"""

import unittest

import paddle
import paddle.nn.clip as clip_module
from paddle.nn.clip import (
    BaseErrorClipAttr,
    ClipGradByGlobalNorm,
    ClipGradByNorm,
    ClipGradByValue,
    ErrorClipByValue,
    _can_inplace_clip_grad,
    _cast_to_mp_type_if_enabled,
    _clip_by_global_norm_using_mp_type,
    _squared_l2_norm,
)


class TestClipByGlobalNormUsingMpType(unittest.TestCase):
    """测试 _clip_by_global_norm_using_mp_type 标志函数
    Test the _clip_by_global_norm_using_mp_type flag function"""

    def setUp(self):
        # 重置全局状态 / Reset global state
        _clip_by_global_norm_using_mp_type(False)

    def tearDown(self):
        # 清理全局状态 / Clean up global state
        _clip_by_global_norm_using_mp_type(False)

    def test_get_flag_without_arg(self):
        """测试无参数时获取标志值
        Test getting flag value without argument"""
        # 默认值为 False / Default value is False
        result = _clip_by_global_norm_using_mp_type()
        self.assertFalse(result)

    def test_set_flag_true(self):
        """测试设置标志为 True
        Test setting flag to True"""
        old = _clip_by_global_norm_using_mp_type(True)
        self.assertFalse(old)  # 返回旧值 / Returns old value
        self.assertTrue(_clip_by_global_norm_using_mp_type())

    def test_set_flag_false(self):
        """测试设置标志为 False
        Test setting flag to False"""
        _clip_by_global_norm_using_mp_type(True)
        old = _clip_by_global_norm_using_mp_type(False)
        self.assertTrue(old)  # 返回旧值 / Returns old value
        self.assertFalse(_clip_by_global_norm_using_mp_type())

    def test_set_flag_same_value(self):
        """测试重复设置相同值
        Test setting same value repeatedly"""
        _clip_by_global_norm_using_mp_type(True)
        old = _clip_by_global_norm_using_mp_type(True)
        self.assertTrue(old)

    def test_assert_len_gt_1(self):
        """测试传入多个参数时抛出异常
        Test assertion error when multiple args passed"""
        with self.assertRaises(AssertionError):
            _clip_by_global_norm_using_mp_type(True, False)

    def test_assert_non_bool(self):
        """测试传入非布尔值时抛出异常
        Test assertion error when non-bool passed"""
        with self.assertRaises(AssertionError):
            _clip_by_global_norm_using_mp_type("true")


class TestCastToMpTypeIfEnabled(unittest.TestCase):
    """测试 _cast_to_mp_type_if_enabled 混合精度类型转换
    Test _cast_to_mp_type_if_enabled mixed precision type casting"""

    def setUp(self):
        _clip_by_global_norm_using_mp_type(False)

    def tearDown(self):
        _clip_by_global_norm_using_mp_type(False)

    def test_fp32_no_cast(self):
        """测试 FP32 张量在标志关闭时不转换
        Test FP32 tensor not cast when flag is off"""
        x = paddle.to_tensor([1.0, 2.0], dtype='float32')
        result = _cast_to_mp_type_if_enabled(x)
        self.assertEqual(result.dtype, paddle.float32)

    def test_fp16_with_flag_off(self):
        """测试 FP16 张量在标志关闭时不转换
        Test FP16 tensor not cast when flag is off"""
        x = paddle.to_tensor([1.0, 2.0], dtype='float16')
        result = _cast_to_mp_type_if_enabled(x)
        self.assertEqual(result.dtype, paddle.float16)

    def test_fp16_with_flag_on(self):
        """测试 FP16 张量在标志开启时转换为 FP32
        Test FP16 tensor cast to FP32 when flag is on"""
        _clip_by_global_norm_using_mp_type(True)
        x = paddle.to_tensor([1.0, 2.0], dtype='float16')
        result = _cast_to_mp_type_if_enabled(x)
        self.assertEqual(result.dtype, paddle.float32)

    def test_bf16_with_flag_on(self):
        """测试 BF16 张量在标志开启时转换为 FP32
        Test BF16 tensor cast to FP32 when flag is on"""
        _clip_by_global_norm_using_mp_type(True)
        x = paddle.to_tensor([1.0, 2.0], dtype='bfloat16')
        result = _cast_to_mp_type_if_enabled(x)
        self.assertEqual(result.dtype, paddle.float32)


class TestCanInplaceClipGrad(unittest.TestCase):
    """测试 _can_inplace_clip_grad 判断梯度是否可原地裁剪
    Test _can_inplace_clip_grad判断梯度是否可原地裁剪"""

    def test_initialized_dense_tensor(self):
        """测试已初始化的稠密张量返回 True
        Test initialized dense tensor returns True"""
        x = paddle.to_tensor([1.0, 2.0])
        grad = paddle.to_tensor([0.1, 0.2])
        result = _can_inplace_clip_grad(grad, x)
        self.assertTrue(result)

    def test_zero_dim_tensor(self):
        """测试 0 维张量返回 False
        Test 0-dim tensor returns False"""
        x = paddle.to_tensor(1.0)
        grad = paddle.to_tensor(0.1)
        result = _can_inplace_clip_grad(grad, x)
        self.assertFalse(result)


class TestSquaredL2NormDynamic(unittest.TestCase):
    """测试 _squared_l2_norm 在动态图模式下的计算
    Test _squared_l2_norm in dynamic mode"""

    def test_float32_norm(self):
        """测试 float32 张量的平方 L2 范数
        Test squared L2 norm for float32 tensor"""
        x = paddle.to_tensor([3.0, 4.0], dtype='float32')
        result = _squared_l2_norm(x)
        # 3^2 + 4^2 = 25
        expected = 25.0
        self.assertAlmostEqual(float(result), expected, places=5)

    def test_float64_norm(self):
        """测试 float64 张量的平方 L2 范数
        Test squared L2 norm for float64 tensor"""
        x = paddle.to_tensor([1.0, 2.0, 3.0], dtype='float64')
        result = _squared_l2_norm(x)
        # 1 + 4 + 9 = 14
        expected = 14.0
        self.assertAlmostEqual(float(result), expected, places=5)


class TestErrorClipByValue(unittest.TestCase):
    """测试 ErrorClipByValue 类
    Test ErrorClipByValue class"""

    def test_init_with_min(self):
        """测试指定 min 值的构造
        Test construction with min specified"""
        clip = ErrorClipByValue(max=1.0, min=-0.5)
        self.assertAlmostEqual(clip.max, 1.0)
        self.assertAlmostEqual(clip.min, -0.5)

    def test_init_without_min(self):
        """测试不指定 min 值时的构造（默认为 -max）
        Test construction without min (defaults to -max)"""
        clip = ErrorClipByValue(max=2.0)
        self.assertAlmostEqual(clip.max, 2.0)
        self.assertAlmostEqual(clip.min, -2.0)

    def test_str_representation(self):
        """测试字符串表示
        Test string representation"""
        clip = ErrorClipByValue(max=1.5, min=-0.5)
        s = str(clip)
        self.assertIn("ByValue", s)
        self.assertIn("-0.500000", s)
        self.assertIn("1.500000", s)

    def test_init_max_coercion_to_float(self):
        """测试 max 参数被强制转为 float
        Test max is coerced to float"""
        clip = ErrorClipByValue(max=1)
        self.assertIsInstance(clip.max, float)

    def test_init_min_coercion_to_float(self):
        """测试 min 参数被强制转为 float
        Test min is coerced to float"""
        clip = ErrorClipByValue(max=1, min=0)
        self.assertIsInstance(clip.min, float)


class TestBaseErrorClipAttr(unittest.TestCase):
    """测试 BaseErrorClipAttr 基类
    Test BaseErrorClipAttr base class"""

    def test_str_not_implemented(self):
        """测试 __str__ 抛出 NotImplementedError
        Test __str__ raises NotImplementedError"""
        base = BaseErrorClipAttr()
        with self.assertRaises(NotImplementedError):
            str(base)

    def test_append_clip_op_not_implemented(self):
        """测试 _append_clip_op 抛出 NotImplementedError
        Test _append_clip_op raises NotImplementedError"""
        base = BaseErrorClipAttr()
        with self.assertRaises(NotImplementedError):
            base._append_clip_op(None, "grad")


class TestClipGradByGlobalNormInternal(unittest.TestCase):
    """测试 ClipGradByGlobalNorm 内部特性
    Test ClipGradByGlobalNorm internal features"""

    def setUp(self):
        _clip_by_global_norm_using_mp_type(False)
        paddle.enable_static()

    def tearDown(self):
        _clip_by_global_norm_using_mp_type(False)
        paddle.disable_static()

    def test_process_context_new_group(self):
        """测试 _process_context 创建新分组上下文
        Test _process_context creates new group context"""
        clip = ClipGradByGlobalNorm(clip_norm=1.0, group_name="test_group")
        context = {}
        x = paddle.static.data(name='x', shape=[2, 3], dtype='float32')
        grad = paddle.zeros([2, 3], dtype='float32')
        clip._process_context(context, x, grad)
        self.assertIn("test_group", context)
        self.assertIn("test_group_clip", context)
        self.assertIn("test_group_clip_value", context)

    def test_process_context_duplicate_group_mismatch(self):
        """测试同一分组 clip_norm 不匹配时抛出异常
        Test mismatched clip_norm for same group raises error"""
        clip1 = ClipGradByGlobalNorm(clip_norm=1.0, group_name="test_group")
        clip2 = ClipGradByGlobalNorm(clip_norm=2.0, group_name="test_group")
        context = {}
        x = paddle.static.data(name='x', shape=[2, 3], dtype='float32')
        grad = paddle.zeros([2, 3], dtype='float32')
        clip1._process_context(context, x, grad)
        with self.assertRaises(ValueError):
            clip2._process_context(context, x, grad)

    def test_process_context_duplicate_group_match(self):
        """测试同一分组 clip_norm 匹配时不抛异常
        Test matching clip_norm for same group does not raise"""
        clip1 = ClipGradByGlobalNorm(clip_norm=1.0, group_name="test_group")
        clip2 = ClipGradByGlobalNorm(clip_norm=1.0, group_name="test_group")
        context = {}
        x = paddle.static.data(name='x', shape=[2, 3], dtype='float32')
        grad = paddle.zeros([2, 3], dtype='float32')
        clip1._process_context(context, x, grad)
        clip2._process_context(context, x, grad)
        self.assertEqual(len(context["test_group"]), 2)

    def test_str_representation(self):
        """测试 __str__ 字符串表示
        Test __str__ string representation"""
        clip = ClipGradByGlobalNorm(clip_norm=1.5)
        s = str(clip)
        self.assertIn("GlobalNorm", s)
        self.assertIn("1.500000", s)

    def test_auto_skip_clip_false(self):
        """测试 auto_skip_clip=False 默认行为
        Test auto_skip_clip=False default behavior"""
        clip = ClipGradByGlobalNorm(clip_norm=1.0, auto_skip_clip=False)
        self.assertFalse(clip.auto_skip_clip)

    def test_auto_skip_clip_true(self):
        """测试 auto_skip_clip=True 设置
        Test auto_skip_clip=True setting"""
        clip = ClipGradByGlobalNorm(clip_norm=1.0, auto_skip_clip=True)
        self.assertTrue(clip.auto_skip_clip)

    def test_auto_skip_clip_assert_non_bool(self):
        """测试 auto_skip_clip 传入非布尔值时抛出异常
        Test non-bool auto_skip_clip raises assertion"""
        with self.assertRaises(AssertionError):
            ClipGradByGlobalNorm(clip_norm=1.0, auto_skip_clip="true")


class TestClipGradByNormInternal(unittest.TestCase):
    """测试 ClipGradByNorm 内部特性
    Test ClipGradByNorm internal features"""

    def setUp(self):
        paddle.enable_static()

    def tearDown(self):
        paddle.disable_static()

    def test_str_representation(self):
        """测试 __str__ 字符串表示
        Test __str__ string representation"""
        clip = ClipGradByNorm(clip_norm=2.5)
        s = str(clip)
        self.assertIn("Norm", s)
        self.assertIn("2.500000", s)

    def test_process_context_pass(self):
        """测试 _process_context 为空操作
        Test _process_context is a no-op"""
        clip = ClipGradByNorm(clip_norm=1.0)
        context = {}
        x = paddle.static.data(name='x', shape=[2, 3], dtype='float32')
        grad = paddle.zeros([2, 3], dtype='float32')
        # 应该不抛异常 / Should not raise
        clip._process_context(context, x, grad)

    def test_create_operators(self):
        """测试 _create_operators 方法
        Test _create_operators method"""
        clip = ClipGradByNorm(clip_norm=1.0)
        x = paddle.static.data(name='x', shape=[2, 3], dtype='float32')
        grad = paddle.zeros([2, 3], dtype='float32')
        param, new_grad = clip._create_operators(x, grad)
        self.assertIs(param, x)
        self.assertIsNotNone(new_grad)


class TestClipGradByValueInternal(unittest.TestCase):
    """测试 ClipGradByValue 内部特性
    Test ClipGradByValue internal features"""

    def setUp(self):
        paddle.enable_static()

    def tearDown(self):
        paddle.disable_static()

    def test_str_representation(self):
        """测试 __str__ 字符串表示
        Test __str__ string representation"""
        clip = ClipGradByValue(max=1.0, min=-0.5)
        s = str(clip)
        self.assertIn("Value", s)
        self.assertIn("1.000000", s)
        self.assertIn("-0.500000", s)

    def test_process_context_pass(self):
        """测试 _process_context 为空操作
        Test _process_context is a no-op"""
        clip = ClipGradByValue(max=1.0)
        context = {}
        x = paddle.static.data(name='x', shape=[2, 3], dtype='float32')
        grad = paddle.zeros([2, 3], dtype='float32')
        clip._process_context(context, x, grad)

    def test_create_operators(self):
        """测试 _create_operators 方法
        Test _create_operators method"""
        clip = ClipGradByValue(max=1.0)
        x = paddle.static.data(name='x', shape=[2, 3], dtype='float32')
        grad = paddle.zeros([2, 3], dtype='float32')
        param, new_grad = clip._create_operators(x, grad)
        self.assertIs(param, x)
        self.assertIsNotNone(new_grad)


class TestAllowPureFp16Bf16Clip(unittest.TestCase):
    """测试 _allow_pure_fp16_global_norm_clip 和 _allow_pure_bf16_global_norm_clip
    Test _allow_pure_fp16/bf16_global_norm_clip flags"""

    def setUp(self):
        # 重置标志 / Reset flags
        clip_module._allow_pure_fp16_global_norm_clip_flag = False
        clip_module._allow_pure_bf16_global_norm_clip_flag = False

    def tearDown(self):
        clip_module._allow_pure_fp16_global_norm_clip_flag = False
        clip_module._allow_pure_bf16_global_norm_clip_flag = False

    def test_fp16_clip_flag_get(self):
        """测试获取 fp16 裁剪标志
        Test getting fp16 clip flag"""
        result = clip_module._allow_pure_fp16_global_norm_clip()
        self.assertFalse(result)

    def test_fp16_clip_flag_set(self):
        """测试设置 fp16 裁剪标志
        Test setting fp16 clip flag"""
        old = clip_module._allow_pure_fp16_global_norm_clip(True)
        self.assertFalse(old)
        self.assertTrue(clip_module._allow_pure_fp16_global_norm_clip())

    def test_bf16_clip_flag_get(self):
        """测试获取 bf16 裁剪标志
        Test getting bf16 clip flag"""
        result = clip_module._allow_pure_bf16_global_norm_clip()
        self.assertFalse(result)

    def test_bf16_clip_flag_set(self):
        """测试设置 bf16 裁剪标志
        Test setting bf16 clip flag"""
        old = clip_module._allow_pure_bf16_global_norm_clip(True)
        self.assertFalse(old)
        self.assertTrue(clip_module._allow_pure_bf16_global_norm_clip())


if __name__ == '__main__':
    unittest.main()
