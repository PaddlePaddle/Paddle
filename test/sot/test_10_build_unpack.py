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

# BUILD_TUPLE_UNPACK (new)
# BUILD_LIST_UNPACK (new)
# BUILD_TUPLE_UNPACK_WITH_CALL (new)
# CALL_FUNCTION_EX (new)
# BUILD_MAP_UNPACK (new)
# LIST_EXTEND (new)
# LIST_TO_TUPLE (new)
# DICT_UPDATE (new)
# DICT_MERGE (new)

from __future__ import annotations

import unittest

from test_case_base import TestCaseBase

import paddle


def build_tuple_unpack(x: tuple[paddle.Tensor], y: tuple[paddle.Tensor]):
    z = (*x, *y)

    return z[0] + 1


def build_list_unpack(x: list[paddle.Tensor], y: list[paddle.Tensor]):
    z = [*x, *y]
    return z[0] + 1


def build_tuple_unpack_with_call(
    x: tuple[paddle.Tensor], y: tuple[paddle.Tensor]
):
    z = build_tuple_unpack_with_call_inner(*x, *y)
    return z[0] + 1


def build_tuple_unpack_with_call_inner(
    a: paddle.Tensor, b: paddle.Tensor, c: paddle.Tensor, d: paddle.Tensor
):
    z = (a, b, c, d)
    return z


def build_map_unpack(x: dict[str, paddle.Tensor], y: dict[str, paddle.Tensor]):
    z = {**x, **y}
    return z["a"] + 1


def build_map_unpack_with_call_inner(
    a: paddle.Tensor, b: paddle.Tensor, c: paddle.Tensor, d: paddle.Tensor
):
    z = {"a": a, "b": b, "c": c, "d": d}
    return z


def build_map_unpack_with_call(
    x: dict[str, paddle.Tensor], y: dict[str, paddle.Tensor]
):
    z = build_map_unpack_with_call_inner(**x, **y)
    return z["a"] + 1


class TestBuildUnpack(TestCaseBase):
    def test_simple(self):
        a = paddle.to_tensor(1)
        b = paddle.to_tensor(2)
        c = paddle.to_tensor(3)
        d = paddle.to_tensor(4)

        self.assert_results(build_tuple_unpack, (a, b), (c, d))
        self.assert_results(build_list_unpack, [a, b], [c, d])
        self.assert_results(build_tuple_unpack_with_call, (a, b), (c, d))
        self.assert_results(
            build_map_unpack, {"a": a, "b": b}, {"c": c, "d": d}
        )
        self.assert_results(
            build_map_unpack_with_call, {"a": a, "b": b}, {"c": c, "d": d}
        )


if __name__ == "__main__":
    unittest.main()
