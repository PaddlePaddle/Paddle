# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from test_case_base import TestCaseBase, test_with_faster_guard

import paddle
from paddle.jit import sot
from paddle.jit.sot import symbolic_translate
from paddle.jit.sot.utils import InnerError, strict_mode_guard


def dict_setitem(x):
    x[0] = 1
    return x[0]


def dict_delitem(x):
    del x[0]
    return x


def dict_delitem_getitem(a):
    b = a[0]
    del a[0]
    b[0] = 1
    return a, b


def dict_nested_1(x):
    x[0][0] = 42
    x[1][0] = x[0][0] + x[0][1]
    x[2] = {1: 2}
    return x


def dict_nested_2(x):
    a = x[0]
    b = x[1]
    del a[0]
    a[1] = b[0]
    a[2] = b[1]
    x[1][0] = 42
    del a[1]
    return a, b


def list_append_int(tensor_x, list_a):
    tensor_x = tensor_x + 1
    list_a.append(12)
    return tensor_x, list_a


def list_append_tensor(tensor_x, list_a):
    tensor_x = tensor_x + 1
    list_a.append(tensor_x)
    return tensor_x, list_a


def list_delitem(list_a):
    del list_a[0]
    return list_a[0]


def list_extend(list_a):
    list_a.extend([1, 2, 3])
    return list_a[0]


def list_nested(list_a):
    inner_list = []
    inner_list.append(list_a)
    inner_list[-1].append(12)
    return 12


def list_insert(list_a):
    list_a.insert(0, 1)
    return list_a[0]


def list_remove(list_a):
    list_a.remove(1)
    return list_a[0]


def list_pop(list_a):
    list_a.pop(0)
    list_a.pop()
    list_a.pop(1)
    return list_a[0]


def list_clear(list_a):
    list_a.clear()
    return list_a


def list_sort(list_a):
    list_a.sort()
    return list_a


def list_reverse(list_a):
    list_a.reverse()
    return list_a


def slice_in_for_loop(x, iter_num=3):
    x = paddle.to_tensor(x)
    a = []

    iter_num = paddle.full(shape=[1], fill_value=iter_num, dtype="int32")

    for i in range(iter_num):
        a.append(x)

    for i in range(iter_num):
        a[i] = x
    out = a[2]
    return out


# TODO: Object SideEffect
class CustomObject:
    def __init__(self):
        self.x = 2
        self.y = paddle.to_tensor(1)

    def object_attr_set2(self, x):
        self.outputs = []
        self.outputs.append(x)
        return self.outputs


@sot.psdb.check_no_breakgraph
def object_attr_set(cus_obj, t):
    """object side effect."""
    t = t + 1
    cus_obj.x = t
    return t, cus_obj.x


def object_attr_breakgraph(cus_obj, t):
    t = t + 1
    sot.psdb.breakgraph()
    cus_obj.x = t
    sot.psdb.breakgraph()
    return t, cus_obj.x


@sot.psdb.check_no_breakgraph
def object_attr_tensor_del(cus_obj):
    del cus_obj.y


@sot.psdb.check_no_breakgraph
def object_attr_int_del(cus_obj):
    del cus_obj.x


def slice_list_after_change(l):
    l.reverse()
    sum = 0
    for i, v in zip(range(2), l[2:]):
        sum += v
    return sum


class TestDictSideEffect(TestCaseBase):
    def test_dict_setitem(self):
        self.assert_results_with_side_effects(
            dict_setitem, {0: paddle.to_tensor(0)}
        )
        self.assert_results_with_side_effects(
            dict_setitem, {0: paddle.to_tensor(1)}
        )

    @test_with_faster_guard
    def test_dict_delitem(self):
        self.assert_results_with_side_effects(
            dict_delitem, {0: paddle.to_tensor(0), 1: paddle.to_tensor(1)}
        )
        self.assert_results_with_side_effects(
            dict_delitem, {0: paddle.to_tensor(1), 2: paddle.to_tensor(2)}
        )

    def test_dict_delitem_getitem(self):
        self.assert_results_with_side_effects(
            dict_delitem_getitem, {0: {0: 1, 1: 2}}
        )

    @test_with_faster_guard
    def test_dict_nested_1(self):
        self.assert_results_with_side_effects(
            dict_nested_1, {0: {0: 1, 1: 2}, 1: {0: 1, 1: 2}}
        )
        self.assert_results_with_side_effects(
            dict_nested_1, {0: {0: 123, 1: 2}, 1: {0: 1, 1: 2}}
        )

    @test_with_faster_guard
    def test_dict_nested_2(self):
        self.assert_results_with_side_effects(
            dict_nested_2, {0: {0: 1, 1: 2}, 1: {0: 1, 1: 2}}
        )
        self.assert_results_with_side_effects(
            dict_nested_2, {0: {0: 123, 1: 2}, 1: {0: 1, 1: 2}}
        )


class TestListSideEffect(TestCaseBase):
    @test_with_faster_guard
    def test_list_append(self):
        self.assert_results_with_side_effects(
            list_append_int, paddle.to_tensor(1), [1, 2, 3]
        )
        self.assert_results_with_side_effects(
            list_append_tensor, paddle.to_tensor(2), [1, 2, 3]
        )

    @test_with_faster_guard
    def test_list_delitem(self):
        self.assert_results_with_side_effects(list_delitem, [1, 2, 3])

    @test_with_faster_guard
    def test_list_extend(self):
        self.assert_results_with_side_effects(
            list_extend, [1, 2, 3, 4, 5, 6, 7, 8, 9]
        )

    @test_with_faster_guard
    def test_list_insert(self):
        self.assert_results_with_side_effects(list_insert, [1, 2, 3])
        self.assert_results_with_side_effects(
            list_insert, [-1, 2, -3, 4, -5, 6, -7, 8, -9]
        )

    @test_with_faster_guard
    def test_list_remove(self):
        self.assert_results_with_side_effects(list_remove, [1, 1, 1])
        self.assert_results_with_side_effects(list_remove, [0, 1, 2])
        with self.assertRaises(InnerError):
            symbolic_translate(list_remove)([0, 2, 4])

    @test_with_faster_guard
    def test_list_pop(self):
        self.assert_results_with_side_effects(list_pop, [1, 2, 3, 4, 5])
        self.assert_results_with_side_effects(
            list_pop, [-1, 2, -3, 4, -5, 6, -7, 8, -9]
        )

    @test_with_faster_guard
    def test_list_clear(self):
        self.assert_results_with_side_effects(list_clear, [1, 2, 3, 4, 5])
        self.assert_results_with_side_effects(
            list_clear, [-1, 2, -3, 4, -5, 6, -7, 8, -9]
        )

    @test_with_faster_guard
    def test_list_sort(self):
        self.assert_results_with_side_effects(list_sort, [2, 1, 7, 3, 4, 6])
        self.assert_results_with_side_effects(
            list_sort, [-1, 2, -3, 4, -5, 6, -7, 8, -9]
        )

    @test_with_faster_guard
    def test_list_reverse(self):
        self.assert_results_with_side_effects(list_reverse, [1, 2, 3, 4, 5])
        self.assert_results_with_side_effects(
            list_reverse, [-1, 2, -3, 4, -5, 6, -7, 8, -9]
        )

    @test_with_faster_guard
    def test_slice_in_for_loop(self):
        x = 2
        with strict_mode_guard(False):
            self.assert_results_with_side_effects(slice_in_for_loop, x)

    @test_with_faster_guard
    def test_list_nested(self):
        self.assert_results_with_side_effects(list_nested, [1, 2, 3])


class TestSliceAfterChange(TestCaseBase):
    @test_with_faster_guard
    def test_slice_list_after_change(self):
        self.assert_results_with_side_effects(
            slice_list_after_change, [1, 2, 3, 4]
        )
        self.assert_results_with_side_effects(
            slice_list_after_change, [7, 8, 9, 10]
        )


class TestAttrSideEffect(TestCaseBase):
    def attr_check(self, func, attr_keys: list[str], cls, *inputs):
        cus_obj1 = cls()
        cus_obj2 = cls()
        sym_output = symbolic_translate(func)(cus_obj1, *inputs)
        paddle_output = func(cus_obj2, *inputs)
        for key in attr_keys:
            self.assert_nest_match(
                getattr(cus_obj1, key, f"__MISS_KEY__{key}"),
                getattr(cus_obj2, key, f"__MISS_KEY__{key}"),
            )
        self.assert_nest_match(sym_output, paddle_output)

    def test_attr_set(self):
        self.attr_check(object_attr_set, ["x"], CustomObject, 5)
        self.attr_check(
            CustomObject.object_attr_set2, ["outputs"], CustomObject, 6
        )
        self.attr_check(
            CustomObject.object_attr_set2,
            ["outputs"],
            CustomObject,
            paddle.to_tensor(5),
        )
        self.attr_check(
            object_attr_set, ["x"], CustomObject, paddle.to_tensor(5)
        )

    def test_attr_del(self):
        self.attr_check(object_attr_tensor_del, ["y"], CustomObject)
        self.attr_check(object_attr_int_del, ["x"], CustomObject)

    def test_attr_set_breakgraph(self):
        self.attr_check(object_attr_breakgraph, ["x"], CustomObject, 100)
        self.attr_check(object_attr_breakgraph, ["x"], CustomObject, 1000)


if __name__ == "__main__":
    unittest.main()
