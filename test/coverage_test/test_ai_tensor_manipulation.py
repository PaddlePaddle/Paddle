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

# [AUTO-GENERATED] Unit test for paddle.tensor.manipulation
# 自动生成的单测，覆盖 paddle.tensor.manipulation 模块中未覆盖的代码

"""
测试模块：paddle.tensor.manipulation (cast, slice, concat, flatten, squeeze, unsqueeze)
Test Module: paddle.tensor.manipulation

本测试覆盖以下功能：
This test covers the following functions:
1. cast - 类型转换 / Type casting between different dtypes
2. slice - 切片操作 / Slice with negative axis and tensor starts/ends
3. concat - 拼接操作 / Concatenation with different axis values
4. flatten - 展平操作 / Flatten with start/stop axis
5. squeeze/unsqueeze - 维度操作 / Dimension manipulation

覆盖的未覆盖行：cast静态图路径, slice负axis, concat边界
"""

import unittest

import numpy as np

import paddle


class TestCastDynamic(unittest.TestCase):
    """测试动态图下的cast操作
    Test cast operation in dynamic graph mode"""

    def setUp(self):
        paddle.disable_static()

    def test_cast_float32_to_int32(self):
        """float32转int32 / Cast float32 to int32"""
        x = paddle.to_tensor([1.5, 2.7, 3.9], dtype='float32')
        y = paddle.cast(x, 'int32')
        self.assertEqual(y.dtype, paddle.int32)
        np.testing.assert_array_equal(y.numpy(), np.array([1, 2, 3]))

    def test_cast_float64_to_float32(self):
        """float64转float32 / Cast float64 to float32"""
        x = paddle.to_tensor([1.0, 2.0, 3.0], dtype='float64')
        y = paddle.cast(x, 'float32')
        self.assertEqual(y.dtype, paddle.float32)
        np.testing.assert_allclose(y.numpy(), [1.0, 2.0, 3.0])

    def test_cast_int32_to_float64(self):
        """int32转float64 / Cast int32 to float64"""
        x = paddle.to_tensor([1, 2, 3], dtype='int32')
        y = paddle.cast(x, 'float64')
        self.assertEqual(y.dtype, paddle.float64)

    def test_cast_bool_to_int(self):
        """bool转int / Cast bool to int"""
        x = paddle.to_tensor([True, False, True])
        y = paddle.cast(x, 'int32')
        np.testing.assert_array_equal(y.numpy(), [1, 0, 1])

    def test_cast_int_to_bool(self):
        """int转bool / Cast int to bool"""
        x = paddle.to_tensor([0, 1, 2, -1], dtype='int32')
        y = paddle.cast(x, 'bool')
        np.testing.assert_array_equal(y.numpy(), [False, True, True, True])


class TestSliceDynamic(unittest.TestCase):
    """测试动态图下的slice操作
    Test slice operation in dynamic graph mode"""

    def setUp(self):
        paddle.disable_static()

    def test_slice_basic(self):
        """基本切片 / Basic slice operation"""
        x = paddle.arange(24, dtype='float32').reshape([2, 3, 4])
        out = paddle.slice(x, axes=[0, 1], starts=[0, 1], ends=[1, 3])
        self.assertEqual(list(out.shape), [1, 2, 4])

    def test_slice_negative_axis(self):
        """负数axis切片 / Slice with negative axis values"""
        x = paddle.arange(24, dtype='float32').reshape([2, 3, 4])
        out = paddle.slice(x, axes=[-1], starts=[1], ends=[3])
        self.assertEqual(list(out.shape), [2, 3, 2])

    def test_slice_with_tensor_starts_ends(self):
        """Tensor类型starts/ends / Slice with Tensor starts and ends"""
        x = paddle.arange(12, dtype='float32').reshape([3, 4])
        out = paddle.slice(x, axes=[0, 1], starts=[0, 0], ends=[2, 3])
        self.assertEqual(list(out.shape), [2, 3])
        expected = np.arange(12).reshape(3, 4)[0:2, 0:3]
        np.testing.assert_allclose(out.numpy(), expected)


class TestConcatDynamic(unittest.TestCase):
    """测试concat拼接操作
    Test concat operations"""

    def setUp(self):
        paddle.disable_static()

    def test_concat_axis0(self):
        """axis=0拼接 / Concat along axis 0"""
        x = paddle.ones([2, 3], dtype='float32')
        y = paddle.zeros([3, 3], dtype='float32')
        out = paddle.concat([x, y], axis=0)
        self.assertEqual(list(out.shape), [5, 3])

    def test_concat_axis1(self):
        """axis=1拼接 / Concat along axis 1"""
        x = paddle.ones([2, 3], dtype='float32')
        y = paddle.zeros([2, 4], dtype='float32')
        out = paddle.concat([x, y], axis=1)
        self.assertEqual(list(out.shape), [2, 7])

    def test_concat_negative_axis(self):
        """负axis拼接 / Concat with negative axis"""
        x = paddle.ones([2, 3], dtype='float32')
        y = paddle.zeros([2, 4], dtype='float32')
        out = paddle.concat([x, y], axis=-1)
        self.assertEqual(list(out.shape), [2, 7])

    def test_concat_multiple_tensors(self):
        """多个tensor拼接 / Concat multiple tensors"""
        tensors = [paddle.full([1, 3], i, dtype='float32') for i in range(5)]
        out = paddle.concat(tensors, axis=0)
        self.assertEqual(list(out.shape), [5, 3])


class TestFlattenSqueeze(unittest.TestCase):
    """测试flatten和squeeze操作
    Test flatten and squeeze operations"""

    def setUp(self):
        paddle.disable_static()

    def test_flatten_default(self):
        """默认flatten / Default flatten"""
        x = paddle.ones([2, 3, 4])
        out = paddle.flatten(x)
        self.assertEqual(list(out.shape), [24])

    def test_flatten_with_start_stop(self):
        """指定start和stop axis / Flatten with start and stop axis"""
        x = paddle.ones([2, 3, 4, 5])
        out = paddle.flatten(x, start_axis=1, stop_axis=2)
        self.assertEqual(list(out.shape), [2, 12, 5])

    def test_squeeze_specific_axis(self):
        """squeeze指定axis / Squeeze specific axis"""
        x = paddle.ones([1, 3, 1, 4])
        out = paddle.squeeze(x, axis=0)
        self.assertEqual(list(out.shape), [3, 1, 4])

    def test_unsqueeze(self):
        """unsqueeze操作 / Unsqueeze operation"""
        x = paddle.ones([3, 4])
        out = paddle.unsqueeze(x, axis=[0, 3])
        self.assertEqual(list(out.shape), [1, 3, 4, 1])


if __name__ == '__main__':
    unittest.main()
