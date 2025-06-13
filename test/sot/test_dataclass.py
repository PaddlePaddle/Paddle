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
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable

from test_case_base import (
    TestCaseBase,
    test_instruction_translator_cache_context,
)

import paddle
from paddle.jit.sot.psdb import check_no_breakgraph
from paddle.jit.sot.utils import strict_mode_guard


@dataclass
class Data:
    x: paddle.Tensor


@dataclass
class DataWithPostInit:
    x: paddle.Tensor

    def __post_init__(self):
        self.x += 1


def return_dataclass(x):
    return Data(x + 1)


def return_dataclass_with_post_init(x):
    return DataWithPostInit(x)


class TestDataclass(TestCaseBase):
    @strict_mode_guard(False)
    def test_dtype_reconstruct(self):
        x = paddle.to_tensor(1)
        self.assert_results(return_dataclass, x)

    @strict_mode_guard(False)
    def test_dtype_reconstruct_with_post_init(self):
        x = paddle.to_tensor(1)
        self.assert_results(return_dataclass_with_post_init, x)


class DataType(IntEnum):
    FLOAT32 = 1
    FLOAT64 = 2
    INT32 = 3
    INT64 = 4


@dataclass
class DataMeta:
    x: paddle.Tensor
    y: paddle.Tensor = None
    z: DataType = DataType.FLOAT32
    m: list[list[paddle.Tensor]] = field(default_factory=list)
    n: int = 0
    f: Callable[[DataMeta], list] = None

    def __post_init__(self):
        self.x += 1


@check_no_breakgraph
def is_eq(data: DataMeta, data2: DataMeta):
    return data == data2


@check_no_breakgraph
def get_attr(data: DataMeta):
    return data.x + data.y


@check_no_breakgraph
def set_attr(data: DataMeta):
    ori_x = data.x
    data.x = data.x + data.n
    res = data.x
    data.x = ori_x
    return res


@check_no_breakgraph
def callable_attr(data: DataMeta):
    return data.f(data)


class TestDataClassInstance(TestCaseBase):
    def test_guard(self):
        d1 = Data(x=paddle.randn([1]))
        dm1 = DataMeta(x=paddle.randn([1]))
        dm2 = DataMeta(x=paddle.randn([1]))
        dm3 = DataMeta(x=paddle.zeros([1]))
        dm4 = DataMeta(x=paddle.randn([1]), z=DataType.INT32)
        dm5 = DataMeta(x=paddle.randn([1]), n=1)
        with test_instruction_translator_cache_context() as ctx:
            self.assertEqual(ctx.translate_count, 0)
            self.assert_results(is_eq, dm1, dm2)
            self.assertEqual(ctx.translate_count, 1)
            self.assert_results(is_eq, dm1, dm2)
            self.assertEqual(ctx.translate_count, 1)
            self.assert_results(is_eq, dm1, d1)
            self.assertEqual(ctx.translate_count, 2)
            self.assert_results(is_eq, dm1, dm3)
            self.assertEqual(ctx.translate_count, 2)
            self.assert_results(is_eq, dm1, dm4)
            self.assertEqual(ctx.translate_count, 3)
            self.assert_results(is_eq, dm1, dm5)
            self.assertEqual(ctx.translate_count, 4)

    def test_get_attr(self):
        dm = DataMeta(x=paddle.randn([1, 2]), y=paddle.randn([1]))
        self.assert_results(get_attr, dm)

    def test_set_attr(self):
        dm = DataMeta(x=paddle.ones([1, 2]), n=2)
        self.assert_results(set_attr, dm)

    def test_callable_attr(self):

        def process_func(data: DataMeta):
            return data.x.shape

        dm = DataMeta(x=paddle.randn([1, 2]), f=process_func)
        self.assert_results(callable_attr, dm)

    def test_eq(self):
        dm1 = DataMeta(x=paddle.randn([1]))
        dm2 = DataMeta(x=paddle.randn([1]))
        dm3 = DataMeta(x=paddle.zeros([1]))
        dm4 = DataMeta(x=paddle.randn([1]), z=DataType.INT32)
        self.assert_results(is_eq, dm1, dm2)
        self.assert_results(is_eq, dm1, dm3)
        self.assert_results(is_eq, dm1, dm4)
        # TODO(wangmingkai): operator.eq with args UserDefinedFunctionVariable
        # dm5 = DataMeta(x= paddle.randn([1]), f=lambda _: [])
        # self.assert_results(is_eq, dm1, dm5)


if __name__ == "__main__":
    unittest.main()
