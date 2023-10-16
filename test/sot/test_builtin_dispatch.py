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

import math
import operator
import unittest
import weakref

from test_case_base import (
    TestCaseBase,
    test_instruction_translator_cache_context,
)

import paddle
from paddle.jit.sot.psdb import check_no_breakgraph


def dispatch_len(x: paddle.Tensor):
    return len(x.shape)


def dispatch_tensor_len(x: paddle.Tensor):
    return len(x)


def dispatch_reversed(x: paddle.Tensor | int, y: paddle.Tensor | int):
    return list(reversed([x + 1, y - 1, x * 10, y + 1000]))


def dispatch_bool(x: paddle.Tensor):
    return operator.truth(x.shape) and bool(x.shape)


def dispatch_ceil(x: paddle.Tensor | float):
    return math.ceil(x) + 1


def dispatch_floor(x: paddle.Tensor | float):
    return math.floor(x) + 1


def test_sum_tuple(x: paddle.Tensor | int, y: paddle.Tensor | int):
    return sum((x, y))


def test_sum_tuple2(
    x: paddle.Tensor | int | list[int] | list[paddle.Tensor],
    y: paddle.Tensor | int | list[int] | list[paddle.Tensor],
):
    return sum((x, y), x)


def test_sum_tuple3(x):
    return sum((), x)


def test_sum_list(x: paddle.Tensor | int, y: paddle.Tensor | int):
    return sum([x, y])


def test_sum_list2(
    x: paddle.Tensor | int | list[int] | list[paddle.Tensor],
    y: paddle.Tensor | int | list[int] | list[paddle.Tensor],
):
    return sum([x, y], x)


def test_sum_list3(x):
    return sum([], x)


def test_tensor_sum(x: paddle.Tensor):
    return sum(x)


def test_tensor_sum_api(x: paddle.Tensor):
    return x.sum()


def test_pow(x: paddle.Tensor | int, y: paddle.Tensor | int):
    return pow(x, y)


def test_pow2(x: paddle.Tensor | int, y: paddle.Tensor | int):
    return pow(x, y, 1)


def test_tensor_pow_api(x: paddle.Tensor, y: paddle.Tensor | int):
    return x.pow(y)


def test_math_pow(x: int, y: int):
    return math.pow(x, y)


def test_chr(x: int | hex | paddle.Tensor):
    return chr(x)


def test_ord(x: str):
    return ord(x)


@check_no_breakgraph
def test_sqrt(x: int):
    return math.sqrt(x)


class TestBuiltinDispatch(TestCaseBase):
    def test_dispatch_len(self):
        self.assert_results(dispatch_len, paddle.to_tensor([1, 2, 3]))

    def test_dispatch_bool(self):
        self.assert_results(dispatch_bool, paddle.to_tensor([1, 2, 3]))

    def test_dispatch_tensor_len(self):
        with test_instruction_translator_cache_context() as ctx:
            self.assert_results(
                dispatch_tensor_len, paddle.to_tensor([1, 2, 3])
            )
            self.assertEqual(ctx.translate_count, 1)
            self.assert_results(
                dispatch_tensor_len, paddle.to_tensor([4, 5, 6])
            )
            self.assertEqual(ctx.translate_count, 1)

    def test_dispatch_list_reversed(self):
        self.assert_results(dispatch_reversed, paddle.to_tensor(1), 2)
        self.assert_results(dispatch_reversed, 2, paddle.to_tensor(1))

    def test_dispatch_tensor_reversed(self):
        self.assert_results(
            dispatch_reversed,
            paddle.to_tensor([1, 2]),
            paddle.to_tensor([3, 4]),
        )

    def test_not_dispatch_tensor_ceil(self):
        # ceil should break graph, since it returns a int rather than a tensor
        self.assert_results(dispatch_ceil, paddle.to_tensor(1.2))

    def test_dispatch_float_ceil(self):
        self.assert_results(dispatch_ceil, 1.2)

    def test_not_dispatch_tensor_floor(self):
        # floor should break graph, since it returns a int rather than a tensor
        self.assert_results(dispatch_floor, paddle.to_tensor(1.2))

    def test_dispatch_float_floor(self):
        self.assert_results(dispatch_floor, 1.2)

    def test_dispatch_sum(self):
        self.assert_results(test_sum_tuple, 1, 1)
        self.assert_results(test_sum_tuple, paddle.to_tensor(1), 1)
        self.assert_results(
            test_sum_tuple, paddle.to_tensor(1), paddle.to_tensor(1)
        )
        self.assert_results(
            test_sum_tuple, paddle.to_tensor([1, 2]), paddle.to_tensor(1)
        )
        self.assert_results(
            test_sum_tuple, paddle.to_tensor([1, 2]), paddle.to_tensor([1, 3])
        )
        self.assert_results(test_sum_tuple2, 1, 1)
        self.assert_results(test_sum_tuple2, [1, 2], [3, 4])
        self.assert_results(test_sum_tuple2, paddle.to_tensor(1), 1)
        self.assert_results(
            test_sum_tuple2, paddle.to_tensor(1), paddle.to_tensor(1)
        )
        self.assert_results(
            test_sum_tuple2,
            [paddle.to_tensor(1), paddle.to_tensor(2)],
            [paddle.to_tensor(3), paddle.to_tensor(4)],
        )
        self.assert_results(
            test_sum_tuple2, paddle.to_tensor([1, 2]), paddle.to_tensor(1)
        )
        self.assert_results(
            test_sum_tuple2, paddle.to_tensor([1, 2]), paddle.to_tensor([1, 3])
        )
        self.assert_results(test_sum_tuple3, 1)
        self.assert_results(test_sum_tuple3, paddle.to_tensor(1))
        self.assert_results(test_sum_list, 1, 1)
        self.assert_results(test_sum_list, paddle.to_tensor(1), 1)
        self.assert_results(
            test_sum_list, paddle.to_tensor(1), paddle.to_tensor(1)
        )
        self.assert_results(
            test_sum_list, paddle.to_tensor([1, 2]), paddle.to_tensor(1)
        )
        self.assert_results(
            test_sum_list, paddle.to_tensor([1, 2]), paddle.to_tensor([1, 3])
        )
        self.assert_results(test_sum_list2, 1, 1)
        self.assert_results(test_sum_list2, [1, 2], [3, 4])
        self.assert_results(test_sum_list2, paddle.to_tensor(1), 1)
        self.assert_results(
            test_sum_list2, paddle.to_tensor(1), paddle.to_tensor(1)
        )
        self.assert_results(
            test_sum_list2,
            [paddle.to_tensor(1), paddle.to_tensor(2)],
            [paddle.to_tensor(3), paddle.to_tensor(4)],
        )
        self.assert_results(
            test_sum_list2, paddle.to_tensor([1, 2]), paddle.to_tensor(1)
        )
        self.assert_results(
            test_sum_list2, paddle.to_tensor([1, 2]), paddle.to_tensor([1, 3])
        )
        self.assert_results(test_sum_list3, 1)
        self.assert_results(test_sum_list3, paddle.to_tensor(1))
        self.assert_results(test_tensor_sum, paddle.to_tensor([1, 2]))
        self.assert_results(test_tensor_sum, paddle.to_tensor((1, 2)))
        self.assert_results(test_tensor_sum_api, paddle.to_tensor([1, 2]))
        self.assert_results(test_tensor_sum_api, paddle.to_tensor((1, 2)))

    def test_dispatch_pow(self):
        self.assert_results(test_pow, 2, 3)
        self.assert_results(test_pow, paddle.to_tensor(2), 3)
        self.assert_results(test_pow, paddle.to_tensor(2), paddle.to_tensor(3))
        self.assert_results(test_pow2, 2, 3)
        self.assert_results(test_math_pow, 2, 3)
        self.assert_results(test_tensor_pow_api, paddle.to_tensor(2), 3)
        self.assert_results(
            test_tensor_pow_api, paddle.to_tensor(2), paddle.to_tensor(3)
        )

    def test_dispatch_chr(self):
        self.assert_results(test_chr, 65)
        self.assert_results(test_chr, 0x41)
        self.assert_results(test_chr, paddle.to_tensor(65))
        self.assert_results(test_chr, paddle.to_tensor(0x41))

    def test_dispatch_ord(self):
        self.assert_results(test_ord, "a")

    def test_dispatch_sqrt(self):
        self.assert_results(test_sqrt, 9)


def run_getattr(x: paddle.Tensor):
    attr = 'dtype'
    out = getattr(x, attr)
    return out


class TestGetattr(TestCaseBase):
    def test_getattr(self):
        x = paddle.to_tensor(4)
        self.assert_results(run_getattr, x)


def tensor_hasattr(x: paddle.Tensor):
    return (
        hasattr(x, "dtype"),
        hasattr(x, "stop_gradient"),
        hasattr(x, "abs"),
        hasattr(x, "non_tensor_attr"),
    )


class ObjectHasattr:
    def __init__(self):
        attr1 = 1
        attr2 = "2"
        attr3 = [3]


def object_hasattr(x: ObjectHasattr):
    return (
        hasattr(x, "attr1"),
        hasattr(x, "attr2"),
        hasattr(x, "attr3"),
        hasattr(x, "non_obj_attr"),
    )


def layer_hasattr(layer: paddle.nn.Layer):
    return (
        hasattr(layer, "parameters"),
        hasattr(layer, "sublayers"),
        hasattr(layer, "non_layer_attr"),
    )


class TestHasattr(TestCaseBase):
    def test_tensor_hasattr(self):
        x = paddle.to_tensor(4)
        self.assert_results(tensor_hasattr, x)

    def test_object_hasattr(self):
        x = ObjectHasattr()
        self.assert_results(object_hasattr, x)

    def test_layer_hasattr(self):
        x = paddle.nn.Layer()
        self.assert_results(layer_hasattr, x)


class WeakrefableObject:
    ...


def weakref_breakgraph(obj):
    return weakref.ref(obj)


class TestWeakref(TestCaseBase):
    def test_weakref_breakgraph(self):
        obj = WeakrefableObject()
        self.assert_results(weakref_breakgraph, obj)


if __name__ == "__main__":
    unittest.main()
