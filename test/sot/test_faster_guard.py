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

from __future__ import annotations

import unittest
from collections import OrderedDict

import numpy as np

import paddle


class TestBasicFasterGuard(unittest.TestCase):
    def test_lambda_guard(self):
        guard_lambda = paddle.framework.core.LambdaGuard(lambda x: x == 1)
        self.assertTrue(guard_lambda.check(1))
        self.assertFalse(guard_lambda.check(2))

    def test_type_match_guard(self):
        guard_int = paddle.framework.core.TypeMatchGuard(int)
        guard_str = paddle.framework.core.TypeMatchGuard(str)
        guard_list = paddle.framework.core.TypeMatchGuard(list)
        self.assertTrue(guard_int.check(1))
        self.assertFalse(guard_int.check("1"))
        self.assertTrue(guard_str.check("1"))
        self.assertFalse(guard_str.check(1))
        self.assertTrue(guard_list.check([1]))
        self.assertFalse(guard_list.check(1))

    def test_isinstance_match_guard(self):
        guard_int = paddle.framework.core.InstanceCheckGuard(int)
        guard_str = paddle.framework.core.InstanceCheckGuard(str)
        guard_list = paddle.framework.core.InstanceCheckGuard(list)
        guard_int_bool = paddle.framework.core.InstanceCheckGuard((int, bool))
        guard_int_str = paddle.framework.core.InstanceCheckGuard((int, str))
        self.assertTrue(guard_int.check(1))
        self.assertFalse(guard_int.check("1"))
        self.assertTrue(guard_str.check("1"))
        self.assertFalse(guard_str.check(1))
        self.assertTrue(guard_list.check([1]))
        self.assertFalse(guard_list.check(1))
        self.assertTrue(guard_int_bool.check(1))
        self.assertTrue(guard_int_bool.check(True))
        self.assertFalse(guard_int_bool.check("1"))
        self.assertTrue(guard_int_str.check(1))
        self.assertTrue(guard_int_str.check("1"))
        self.assertTrue(guard_int_str.check(True))
        self.assertFalse(guard_int_str.check([1]))

    def test_value_match_guard(self):
        guard_value = paddle.framework.core.ValueMatchGuard(1)
        guard_container_value = paddle.framework.core.ValueMatchGuard([1])
        self.assertTrue(guard_value.check(1))
        self.assertFalse(guard_value.check(2))
        self.assertTrue(guard_container_value.check([1]))
        self.assertFalse(guard_container_value.check([2]))

    def test_length_match_guard(self):
        guard_length = paddle.framework.core.LengthMatchGuard(1)
        # list
        self.assertTrue(guard_length.check([1]))
        self.assertFalse(guard_length.check([1, 2]))

        # dict
        order_dict = OrderedDict()
        order_dict[1] = 2
        self.assertTrue(guard_length.check(order_dict))
        self.assertTrue(guard_length.check({1: 2}))

        # tuple
        self.assertTrue(guard_length.check((1,)))
        self.assertFalse(guard_length.check((1, 2)))

    def test_dtype_match_guard(self):
        guard_dtype = paddle.framework.core.DtypeMatchGuard(paddle.int32)
        self.assertTrue(
            guard_dtype.check(paddle.to_tensor(1, dtype=paddle.int32))
        )
        self.assertFalse(
            guard_dtype.check(paddle.to_tensor(1, dtype=paddle.float32))
        )

    def test_shape_match_guard(self):
        tensor = paddle.randn([2, 3])
        guard_shape = paddle.framework.core.ShapeMatchGuard([2, 3])
        self.assertTrue(guard_shape.check(tensor))
        guard_shape = paddle.framework.core.ShapeMatchGuard([2, None])
        self.assertTrue(guard_shape.check(tensor))
        guard_shape = paddle.framework.core.ShapeMatchGuard([3, 2])
        self.assertFalse(guard_shape.check(tensor))
        guard_shape = paddle.framework.core.ShapeMatchGuard([2, 3, 1])
        self.assertFalse(guard_shape.check(tensor))

    def test_attribute_match_guard(self):
        a = range(1, 10, 2)
        guard_attribute = paddle.framework.core.AttributeMatchGuard(a, "start")
        self.assertTrue(guard_attribute.check(a))
        self.assertFalse(guard_attribute.check(range(10)))

    def test_layer_match_guard(self):
        layer = paddle.nn.Linear(10, 10)
        guard_layer = paddle.framework.core.LayerMatchGuard(layer)
        self.assertTrue(guard_layer.check(layer))
        self.assertFalse(guard_layer.check(paddle.nn.Linear(10, 10)))
        layer.eval()
        self.assertFalse(guard_layer.check(layer))
        layer.train()
        self.assertTrue(guard_layer.check(layer))

    def test_id_match_guard(self):
        layer = paddle.nn.Linear(10, 10)
        guard_id = paddle.framework.core.IdMatchGuard(layer)
        self.assertTrue(guard_id.check(layer))
        layer.eval()
        self.assertTrue(guard_id.check(layer))
        self.assertFalse(guard_id.check(paddle.nn.Linear(10, 10)))

    def test_numpy_dtype_match_guard(self):
        np_array = np.array(1, dtype=np.int32)
        guard_numpy_dtype = paddle.framework.core.NumPyDtypeMatchGuard(
            np_array.dtype
        )
        self.assertTrue(guard_numpy_dtype.check(np_array))
        self.assertTrue(guard_numpy_dtype.check(np.array(1, dtype=np.int32)))
        self.assertTrue(guard_numpy_dtype.check(np.int32()))
        self.assertFalse(guard_numpy_dtype.check(np.array(1, dtype=np.int64)))
        self.assertFalse(guard_numpy_dtype.check(np.float32()))
        self.assertFalse(guard_numpy_dtype.check(np.bool_()))

        np_bool = np.bool_(1)
        guard_numpy_bool_dtype = paddle.framework.core.NumPyDtypeMatchGuard(
            np_bool.dtype
        )
        self.assertTrue(guard_numpy_bool_dtype.check(np.bool_()))
        self.assertTrue(
            guard_numpy_bool_dtype.check(np.array(1, dtype=np.bool_))
        )

    def test_numpy_array_match_guard(self):
        np_array = paddle.framework.core.NumPyArrayValueMatchGuard(
            np.array([1, 2, 3])
        )
        self.assertTrue(np_array.check(np.array([1, 2, 3])))
        self.assertFalse(np_array.check(np.array([4, 5, 6])))

        np_array_all_one = paddle.framework.core.NumPyArrayValueMatchGuard(
            np.array([1, 1, 1])
        )
        self.assertTrue(np_array_all_one.check(np.array([1, 1, 1])))
        self.assertFalse(np_array_all_one.check(np.array([1, 2, 3])))
        self.assertTrue(np_array_all_one.check(np.array([1, 1, 1], dtype=bool)))
        self.assertTrue(np_array_all_one.check(np.array([True, True, True])))
        self.assertTrue(np_array_all_one.check(np.array([1, 1, 1], dtype=int)))

        np_bool_array = paddle.framework.core.NumPyArrayValueMatchGuard(
            np.array([True, False, True])
        )
        self.assertTrue(np_bool_array.check(np.array([True, False, True])))
        self.assertFalse(np_bool_array.check(np.array([True, True, True])))
        self.assertFalse(np_bool_array.check(np.array([False, False, False])))
        self.assertFalse(np_bool_array.check(np.array([1, 2, 3])))
        self.assertTrue(np_bool_array.check(np.array([True, False, 1])))
        self.assertFalse(np_bool_array.check(np.array([1, 2, 3], dtype=bool)))
        self.assertFalse(np_bool_array.check(np.array([1, 2, 3], dtype=int)))

        np_bool_array_all_true = (
            paddle.framework.core.NumPyArrayValueMatchGuard(
                np.array([True, True, True])
            )
        )
        self.assertTrue(
            np_bool_array_all_true.check(np.array([True, True, True]))
        )
        self.assertFalse(
            np_bool_array_all_true.check(np.array([True, False, True]))
        )
        self.assertFalse(
            np_bool_array_all_true.check(np.array([False, False, False]))
        )
        self.assertFalse(np_bool_array_all_true.check(np.array([1, 2, 3])))
        self.assertTrue(np_bool_array_all_true.check(np.array([True, True, 1])))

    def test_object_match_guard(self):
        def test_func():
            return 1 + 1

        guard_object = paddle.framework.core.WeakRefMatchGuard(test_func)
        self.assertTrue(guard_object.check(test_func))
        self.assertFalse(guard_object.check(lambda x: x == 1))
        self.assertFalse(guard_object.check(1))
        self.assertFalse(guard_object.check("1"))


class TestFasterGuardGroup(unittest.TestCase):
    def test_guard_group(self):
        guard_lambda = paddle.framework.core.LambdaGuard(lambda x: x == 1)
        guard_type_match = paddle.framework.core.TypeMatchGuard(int)
        guard_group = paddle.framework.core.GuardGroup(
            [guard_type_match, guard_lambda] * 10
        )
        self.assertTrue(guard_group.check(1))
        self.assertFalse(guard_group.check(2))

    def test_nested_guard_group(self):
        guard_lambda = paddle.framework.core.LambdaGuard(lambda x: x == 1)
        guard_type_match = paddle.framework.core.TypeMatchGuard(int)
        guard_group = paddle.framework.core.GuardGroup(
            [guard_type_match, guard_lambda]
        )
        for _ in range(10):
            guard_group = paddle.framework.core.GuardGroup(
                [guard_group, guard_type_match, guard_lambda]
            )
        self.assertTrue(guard_group.check(1))
        self.assertFalse(guard_group.check(2))


if __name__ == "__main__":
    unittest.main()
