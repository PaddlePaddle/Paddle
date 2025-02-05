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
# BUILD_TUPLE
# BINARY_SUBSCR

from __future__ import annotations

import unittest

from test_case_base import TestCaseBase

import paddle
from paddle.jit.sot.psdb import check_no_breakgraph
from paddle.jit.sot.utils import with_control_flow_guard


@check_no_breakgraph
def build_tuple(x: int, y: paddle.Tensor):
    x = (x, y)
    return x[1] + 1


@check_no_breakgraph
def build_tuple_with_slice_subscript(x: int, y: paddle.Tensor):
    z = (x, y, 3, 4)
    return z[0:5:1]


@check_no_breakgraph
def build_tuple_with_int_subscript(x: int, y: paddle.Tensor):
    z = (x, y)
    return z[0]


@check_no_breakgraph
def tuple_count_int(x: int, y: paddle.Tensor):
    z = (x, x, 2, 1)
    return z.count(x)


def tuple_count_tensor(x: paddle.Tensor, y: tuple[paddle.Tensor]):
    return y.count(x)


@check_no_breakgraph
def tuple_index_int(x: int, y: paddle.Tensor):
    z = (x, y, x, y, y)
    return z.index(x)


def tuple_index_tensor(x: paddle.Tensor, y: tuple[paddle.Tensor]):
    return y.index(x)


@check_no_breakgraph
def tuple_compare():
    # TODO(SigureMo): support gt, ge, lt, le
    l1 = (1, 2, 3)
    l2 = (1, 2, 3)
    l3 = (1, 2, 4)
    return l1 == l2, l1 == l3, l1 != l2, l1 != l3


@check_no_breakgraph
def tuple_add():
    l0 = (1, 2, 3)
    l1 = (4, 5, 6)
    return l0 + l1


@check_no_breakgraph
def tuple_inplace_add():
    l0 = (1, 2, 3)
    l1 = l0
    l2 = (4, 5, 6)
    l0 += l2
    return l0, l1


class TestBuildTuple(TestCaseBase):
    def test_build_tuple(self):
        self.assert_results(build_tuple, 1, paddle.to_tensor(2))
        self.assert_results(
            build_tuple_with_slice_subscript, 1, paddle.to_tensor(2)
        )
        self.assert_results(
            build_tuple_with_int_subscript, 1, paddle.to_tensor(2)
        )


class TestTupleMethods(TestCaseBase):
    def test_tuple_methods_int(self):
        self.assert_results(tuple_count_int, 1, paddle.to_tensor(2))
        self.assert_results(tuple_index_int, 1, paddle.to_tensor(2))

    @with_control_flow_guard(False)
    def test_tuple_methods_tensor(self):
        a = paddle.to_tensor(1)
        b = paddle.to_tensor(2)
        self.assert_results(tuple_count_tensor, a, (a, b, a, b))
        self.assert_results(tuple_index_tensor, b, (b, b, b, a))

    def test_tuple_compare(self):
        self.assert_results(tuple_compare)

    def test_tuple_add(self):
        self.assert_results(tuple_add)

    def test_tuple_inplace_add(self):
        self.assert_results(tuple_inplace_add)


if __name__ == "__main__":
    unittest.main()
