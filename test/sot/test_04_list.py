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

# New Supported Instructions:
# BUILD_LIST (new)
# BINARY_SUBSCR
# DELETE_SUBSCR

from __future__ import annotations

import unittest

from test_case_base import TestCaseBase, test_with_faster_guard

import paddle
from paddle.jit.sot.psdb import check_no_breakgraph


@check_no_breakgraph
def list_getitem_int(x: int, y: paddle.Tensor):
    x = [x, y]
    return x[0] + 1


@check_no_breakgraph
def list_getitem_tensor(x: int, y: paddle.Tensor):
    x = [x, y]
    return x[1] + 1


@check_no_breakgraph
def list_setitem_int(x: int, y: paddle.Tensor):
    z = [x, y]
    z[0] = 3
    return z


def list_setitem_tensor(x: int, y: paddle.Tensor):
    z = [x, y]
    z[1] = paddle.to_tensor(3)
    return z


@check_no_breakgraph
def list_delitem_int(x: int, y: paddle.Tensor):
    z = [x, y]
    del z[0]
    return z


@check_no_breakgraph
def list_delitem_tensor(x: int, y: paddle.Tensor):
    z = [x, y]
    del z[1]
    return z


@check_no_breakgraph
def list_construct_from_list(x: int, y: paddle.Tensor):
    z = [x, y]
    return z


@check_no_breakgraph
def list_append_int(x: int, y: paddle.Tensor):
    z = [x, y]
    z.append(3)
    return z


@check_no_breakgraph
def list_append_tensor(x: int, y: paddle.Tensor):
    z = [x, y]
    z.append(y)
    return z


@check_no_breakgraph
def list_clear(x: int, y: paddle.Tensor):
    z = [x, y]
    z.clear()
    return z


@check_no_breakgraph
def list_copy(x: int, y: paddle.Tensor):
    z = [x, y]
    a = z.copy()
    z[0] = 3
    z[1] = y + 1
    return (a, z)


@check_no_breakgraph
def list_count_int(x: int, y: paddle.Tensor):
    z = [x, x, 2, 3, 1]
    return z.count(x)


def list_count_tensor(x: paddle.Tensor, y: list[paddle.Tensor]):
    return y.count(x)


@check_no_breakgraph
def list_extend(x: int, y: paddle.Tensor):
    z = [x, y]
    a = [y, x]
    b = (x, y)
    z.extend(a)
    z.extend(b)
    return z


@check_no_breakgraph
def list_index_int(x: int, y: paddle.Tensor):
    z = [x, x, 1, 2]
    return z.index(x)


def list_index_tensor(x: paddle.Tensor, y: list[paddle.Tensor]):
    return y.index(x)


@check_no_breakgraph
def list_insert(x: int, y: paddle.Tensor):
    z = [x, y]
    z.insert(0, x)
    z.insert(3, y)
    return z


@check_no_breakgraph
def list_pop(x: int, y: paddle.Tensor):
    z = [x, y]
    a = z.pop()
    b = z.pop()
    return (z, a, b)


@check_no_breakgraph
def list_remove(x: int, y: paddle.Tensor):
    z = [x, x, y, y]
    z.remove(x)
    z.remove(y)
    return z


@check_no_breakgraph
def list_reverse(x: int, y: paddle.Tensor):
    z = [x, x, y, y]
    z.reverse()
    return z


@check_no_breakgraph
def list_default_sort(x: int, y: paddle.Tensor):
    z = [x + 2, x, x + 1]
    z.sort()
    return z


@check_no_breakgraph
def list_key_sort(x: int, y: paddle.Tensor):
    z = [x + 2, x, x + 1]
    z.sort(lambda x: x)
    return z


@check_no_breakgraph
def list_reverse_sort(x: int, y: paddle.Tensor):
    z = [x + 2, x, x + 1]
    z.sort(reverse=True)
    return z


@check_no_breakgraph
def list_tensor_sort(x: int, y: paddle.Tensor):
    z = [y + 2, y, y + 1]
    z.sort()
    return z


@check_no_breakgraph
def list_max(x: paddle.Tensor | int, y: paddle.Tensor | int):
    z = [x, x, y]
    return max(z)


@check_no_breakgraph
def list_tensor_max_api(x: paddle.Tensor):
    return x.max()


@check_no_breakgraph
def list_min(x: paddle.Tensor | int, y: paddle.Tensor | int):
    z = [x, x, y]
    return min(z)


@check_no_breakgraph
def list_tensor_min_api(x: paddle.Tensor):
    return x.min()


@check_no_breakgraph
def list_no_arguments():
    l1 = list()  # noqa: C408
    l1.append(1)
    l2 = list()  # noqa: C408
    l2.append(2)
    return l1[0] + l2[0]


@check_no_breakgraph
def list_compare():
    # TODO(SigureMo): support gt, ge, lt, le
    l1 = [1, 2, 3]
    l2 = [1, 2, 3]
    l3 = [1, 2, 4]
    return l1 == l2, l1 == l3, l1 != l2, l1 != l3


@check_no_breakgraph
def list_add():
    l0 = [1, 2, 3]
    l1 = [4, 5, 6]
    return l0 + l1


@check_no_breakgraph
def list_inplace_add():
    l0 = [1, 2, 3]
    l1 = l0
    l2 = [4, 5, 6]
    l0 += l2
    return l0, l1


@check_no_breakgraph
def list_extend_range(x):
    return [1, *range(0, len(x.shape))]


@check_no_breakgraph
def list_extend_dict():
    l1 = []
    l1.extend({1: 2, 2: 3, 3: 4})
    return l1


class TestListBasic(TestCaseBase):
    def test_list_basic(self):
        self.assert_results(list_getitem_int, 1, paddle.to_tensor(2))
        self.assert_results(list_getitem_tensor, 1, paddle.to_tensor(2))
        self.assert_results_with_side_effects(
            list_setitem_int, 1, paddle.to_tensor(2)
        )


class TestListMethods(TestCaseBase):
    def test_list_setitem(self):
        self.assert_results_with_side_effects(
            list_setitem_tensor, 1, paddle.to_tensor(2)
        )

    def test_list_count_and_index(self):
        self.assert_results(list_count_int, 1, paddle.to_tensor(2))
        self.assert_results(list_index_int, 1, paddle.to_tensor(2))
        a = paddle.to_tensor(1)
        b = paddle.to_tensor(2)
        self.assert_results(list_count_tensor, a, [a, b, a, b, a, b])
        self.assert_results(list_index_tensor, b, [a, b, a, b, a, b])

    def test_list_delitem(self):
        self.assert_results_with_side_effects(
            list_delitem_int, 1, paddle.to_tensor(2)
        )
        self.assert_results_with_side_effects(
            list_delitem_tensor, 1, paddle.to_tensor(2)
        )

    def test_list_append(self):
        self.assert_results_with_side_effects(
            list_append_int, 1, paddle.to_tensor(2)
        )
        self.assert_results_with_side_effects(
            list_append_tensor, 1, paddle.to_tensor(2)
        )

    def test_list_clear(self):
        self.assert_results_with_side_effects(
            list_clear, 1, paddle.to_tensor(2)
        )

    def test_list_copy(self):
        self.assert_results_with_side_effects(list_copy, 1, paddle.to_tensor(2))

    def test_list_extend(self):
        self.assert_results_with_side_effects(
            list_extend, 1, paddle.to_tensor(2)
        )

    def test_list_insert(self):
        self.assert_results_with_side_effects(
            list_insert, 1, paddle.to_tensor(2)
        )

    def test_list_pop(self):
        self.assert_results_with_side_effects(list_pop, 1, paddle.to_tensor(2))

    def test_list_remove(self):
        self.assert_results_with_side_effects(
            list_remove, 1, paddle.to_tensor(2)
        )

    def test_list_reverse(self):
        self.assert_results_with_side_effects(
            list_reverse, 1, paddle.to_tensor(2)
        )
        self.assert_results_with_side_effects(
            list_reverse, 1, paddle.to_tensor(2)
        )

    def test_list_sort(self):
        self.assert_results_with_side_effects(
            list_default_sort, 1, paddle.to_tensor(2)
        )
        # TODO: Not currently supported
        # self.assert_results_with_side_effects(
        #     list_tensor_sort, 1, paddle.to_tensor(2)
        # )
        # self.assert_results_with_side_effects(
        #     list_key_sort, 1, paddle.to_tensor(2)
        # )
        # self.assert_results_with_side_effects(
        #     list_reverse_sort, 1, paddle.to_tensor(2)
        # )

    def test_list_construct_from_list(self):
        self.assert_results(list_construct_from_list, 1, paddle.to_tensor(2))

    def test_list_max_min(self):
        self.assert_results(list_max, 1, 2)
        self.assert_results(list_min, 1, 2)
        self.assert_results(list_tensor_max_api, paddle.to_tensor([1, 2, 3]))
        self.assert_results(list_tensor_min_api, paddle.to_tensor([1, 2, 3]))

    def test_list_noargs(self):
        self.assert_results(list_no_arguments)

    def test_list_compare(self):
        self.assert_results(list_compare)

    def test_list_add(self):
        self.assert_results(list_add)

    def test_list_inplace_add(self):
        self.assert_results(list_inplace_add)

    @test_with_faster_guard
    def test_list_extend_range(self):
        self.assert_results(list_extend_range, paddle.to_tensor([1, 2]))

    def test_list_extend_dict(self):
        self.assert_results(list_extend_dict)


if __name__ == "__main__":
    unittest.main()
