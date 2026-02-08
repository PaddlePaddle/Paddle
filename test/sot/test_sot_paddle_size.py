# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from test_case_base import TestCaseBase

import paddle
from paddle.jit.sot.psdb import check_no_breakgraph


@check_no_breakgraph
def size_construct_from_list(x: int, y: int):
    s = paddle.Size([x, y, 3])
    return s


@check_no_breakgraph
def size_construct_from_tuple(x: int, y: int):
    s = paddle.Size((x, y, 3))
    return s


@check_no_breakgraph
def size_getitem_int(x: int, y: int):
    s = paddle.Size([x, y, 10])
    return s[1]


@check_no_breakgraph
def size_getitem_slice(x: int, y: int):
    s = paddle.Size([x, y, 10, 20])
    # Slice of Size should return Size
    return s[1:3]


@check_no_breakgraph
def size_numel(x: int, y: int):
    s = paddle.Size([x, y, 2])
    return s.numel()


# --- Add Operations ---


@check_no_breakgraph
def size_add_list(x: int):
    s = paddle.Size([x, 2])
    l = [3, 4]
    return s + l, l + s


@check_no_breakgraph
def size_add_tuple(x: int):
    s = paddle.Size([x, 2])
    t = (3, 4)
    return s + t, t + s


@check_no_breakgraph
def size_add_size(x: int):
    s1 = paddle.Size([x, 2])
    s2 = paddle.Size([3, 4])
    res = s1 + s2
    return res


# --- Mul Operations ---


@check_no_breakgraph
def size_mul(x: int):
    s = paddle.Size([x, 2])
    res = s * 2
    return res


@check_no_breakgraph
def size_rmul(x: int):
    s = paddle.Size([x, 2])
    res = 2 * s
    return res


# --- Compare Operations ---


@check_no_breakgraph
def size_compare_tuple(x: int):
    s = paddle.Size([x, 2])
    t = (x, 2)
    t_diff = (x, 3, 4)
    return (
        s == t,
        s == t_diff,
        s != t,
        s != t_diff,
        t == s,
        t_diff == s,
        t != s,
        t_diff != s,
    )


@check_no_breakgraph
def size_compare_list(x: int):
    s = paddle.Size([x, 2])
    l = [x, 2]
    return s == l, s != l, l == s, l != s


# --- Common Methods ---


@check_no_breakgraph
def size_count(x: int):
    s = paddle.Size([x, x, 2, 3])
    return s.count(x)


@check_no_breakgraph
def size_index(x: int):
    s = paddle.Size([2, x, 3])
    return s.index(x)


@check_no_breakgraph
def size_len(x: int):
    s = paddle.Size([x, x, 2])
    return len(s)


@check_no_breakgraph
def size_iter(x: int):
    s = paddle.Size([x, 2, 3])
    res = 0
    for dim in s:
        res += dim
    return res


@check_no_breakgraph
def size_contains(x: int):
    s = paddle.Size([x, 2, 3])
    return x in s, 99 in s


# --- Symbolic shape---
@check_no_breakgraph
def symbolic_shape(x: int):
    s = paddle.zeros(shape=[x, 2, 3])
    s[0] = 1
    res = s.unique(return_inverse=False)
    return res.shape, res.shape.numel()


class TestSizeBasic(TestCaseBase):
    def test_construct(self):
        self.assert_results(size_construct_from_list, 1, 2)
        self.assert_results(size_construct_from_tuple, 1, 2)

    def test_getitem(self):
        self.assert_results(size_getitem_int, 5, 6)
        self.assert_results(size_getitem_slice, 5, 6)

    def test_numel(self):
        self.assert_results(size_numel, 3, 4)
        self.assert_results(size_numel, 0, 4)

    def test_add(self):
        self.assert_results(size_add_list, 1)
        self.assert_results(size_add_tuple, 1)
        self.assert_results(size_add_size, 1)

    def test_mul(self):
        self.assert_results(size_mul, 1)
        self.assert_results(size_rmul, 1)

    def test_compare(self):
        self.assert_results(size_compare_tuple, 1)
        self.assert_results(size_compare_list, 1)

    def test_methods(self):
        self.assert_results(size_count, 1)
        self.assert_results(size_index, 1)
        self.assert_results(size_len, 5)
        self.assert_results(size_iter, 4)
        self.assert_results(size_contains, 1)

    def test_symbolic_shape(self):
        self.assert_results(symbolic_shape, 3)


if __name__ == "__main__":
    unittest.main()
