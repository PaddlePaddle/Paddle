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

# [AUTO-GENERATED] Test file for paddle.distributed.communication.serialization_utils
# 覆盖模块: paddle/distributed/communication/serialization_utils.py
# 未覆盖行: 23,24,25,26,27,28,29,32,33,34
# Covered module: paddle/distributed/communication/serialization_utils.py
# Uncovered lines: 23,24,25,26,27,28,29,32,33,34

import unittest

import numpy as np

import paddle
from paddle.distributed.communication.serialization_utils import (
    convert_object_to_tensor,
    convert_tensor_to_object,
)


class TestConvertObjectToTensor(unittest.TestCase):
    """测试 convert_object_to_tensor 函数
    Test convert_object_to_tensor function"""

    def test_convert_dict(self):
        """测试转换字典对象
        Test converting dictionary object"""
        obj = {"key": "value", "num": 42}
        tensor, length = convert_object_to_tensor(obj)
        self.assertIsInstance(tensor, paddle.Tensor)
        self.assertGreater(length, 0)
        self.assertEqual(tensor.dtype, paddle.uint8)

    def test_convert_list(self):
        """测试转换列表对象
        Test converting list object"""
        obj = [1, 2, 3, "hello"]
        tensor, length = convert_object_to_tensor(obj)
        self.assertIsInstance(tensor, paddle.Tensor)
        self.assertEqual(length, tensor.numel())

    def test_convert_string(self):
        """测试转换字符串对象
        Test converting string object"""
        obj = "hello world"
        tensor, length = convert_object_to_tensor(obj)
        self.assertGreater(length, 0)

    def test_convert_nested(self):
        """测试转换嵌套对象
        Test converting nested object"""
        obj = {"a": [1, 2], "b": {"c": 3}}
        tensor, length = convert_object_to_tensor(obj)
        self.assertIsInstance(tensor, paddle.Tensor)
        self.assertGreater(length, 0)

    def test_convert_number(self):
        """测试转换数字对象
        Test converting number object"""
        obj = 3.14159
        tensor, length = convert_object_to_tensor(obj)
        self.assertGreater(length, 0)


class TestConvertTensorToObject(unittest.TestCase):
    """测试 convert_tensor_to_object 函数
    Test convert_tensor_to_object function"""

    def test_roundtrip_dict(self):
        """测试字典的序列化-反序列化循环
        Test dict serialization-deserialization round-trip"""
        original = {"key": "value", "num": 42}
        tensor, length = convert_object_to_tensor(original)
        result = convert_tensor_to_object(tensor, length)
        self.assertEqual(result, original)

    def test_roundtrip_list(self):
        """测试列表的序列化-反序列化循环
        Test list serialization-deserialization round-trip"""
        original = [1, 2, 3, "hello"]
        tensor, length = convert_object_to_tensor(original)
        result = convert_tensor_to_object(tensor, length)
        self.assertEqual(result, original)

    def test_roundtrip_string(self):
        """测试字符串的序列化-反序列化循环
        Test string serialization-deserialization round-trip"""
        original = "hello world"
        tensor, length = convert_object_to_tensor(original)
        result = convert_tensor_to_object(tensor, length)
        self.assertEqual(result, original)

    def test_roundtrip_nested(self):
        """测试嵌套对象的序列化-反序列化循环
        Test nested object serialization-deserialization round-trip"""
        original = {"a": [1, 2], "b": {"c": 3}}
        tensor, length = convert_object_to_tensor(original)
        result = convert_tensor_to_object(tensor, length)
        self.assertEqual(result, original)

    def test_roundtrip_numpy_array(self):
        """测试 numpy 数组的序列化-反序列化循环
        Test numpy array serialization-deserialization round-trip"""
        original = {"arr": np.array([1.0, 2.0, 3.0])}
        tensor, length = convert_object_to_tensor(original)
        result = convert_tensor_to_object(tensor, length)
        np.testing.assert_array_equal(result["arr"], original["arr"])


if __name__ == '__main__':
    unittest.main()
