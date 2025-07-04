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

import sys
import unittest
from dataclasses import dataclass, field

from test_case_base import (
    TestCaseBase,
)

import paddle
from paddle.jit.sot.psdb import check_no_breakgraph


@dataclass
class DataTensor:
    x: paddle.Tensor


@dataclass
class DataInt:
    x: int


@dataclass
class DataTensorWithPostInit:
    x: paddle.Tensor

    def __post_init__(self):
        self.x += 1


def return_dataclass(x):
    return DataTensor(x + 1)


def return_dataclass_with_post_init(x):
    return DataTensorWithPostInit(x)


class TestDataclassBasic(TestCaseBase):
    def test_dtype_reconstruct(self):
        x = paddle.to_tensor(1)
        self.assert_results(return_dataclass, x)

    def test_dtype_reconstruct_with_post_init(self):
        x = paddle.to_tensor(1)
        self.assert_results(return_dataclass_with_post_init, x)


@dataclass
class DataMeta:
    x: paddle.Tensor
    y: paddle.Tensor | None = None
    m: list[list[paddle.Tensor]] = field(default_factory=list)
    n: int = 0

    def __post_init__(self):
        self.x += 1


@check_no_breakgraph
def is_data_int_eq(data1: DataInt, data2: DataInt):
    return data1 == data2


def is_data_tensor_eq(data1: DataTensor, data2: DataTensor):
    return data1 == data2


def is_any_eq(data1, data2):
    return data1 == data2


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
def get__dataclass_fields__(data: DataMeta):
    return list(data.__dataclass_fields__), list(
        data.__class__.__dataclass_fields__
    )


class TestDataClassInstance(TestCaseBase):
    def test_get_attr(self):
        dm = DataMeta(x=paddle.randn([1, 2]), y=paddle.randn([1]))
        self.assert_results(get_attr, dm)
        self.assert_results(get__dataclass_fields__, dm)

    def test_set_attr(self):
        dm = DataMeta(x=paddle.ones([1, 2]), n=2)
        self.assert_results(set_attr, dm)

    def test_eq_int(self):
        di1 = DataInt(x=1)
        di2 = DataInt(x=1)
        di3 = DataInt(x=2)
        self.assert_results(is_data_int_eq, di1, di2)
        self.assert_results(is_data_int_eq, di1, di3)

    def test_eq_tensor(self):
        t = paddle.randn([1])
        dt1 = DataTensor(x=t)
        dt2 = DataTensor(x=t)
        dt3 = DataTensor(x=paddle.zeros([1]))
        self.assert_results(is_data_tensor_eq, dt1, dt2)
        self.assert_results(is_data_tensor_eq, dt1, dt3)

    def test_eq_diff_dataclass(self):
        di = DataInt(x=1)
        dt = DataTensor(x=1)  # type: ignore
        self.assert_results(is_data_int_eq, di, dt)


@dataclass
class ComplexDataClass:
    a: int
    b: int = 0
    c: int = field(default=1)
    d: int = (
        field(default_factory=lambda: 2, kw_only=True)
        if sys.version_info >= (3, 10)
        else field(default_factory=lambda: 2)
    )


def create_dataclass_with_a():
    return ComplexDataClass(0)


def create_dataclass_with_kwarg_a():
    return ComplexDataClass(a=1)


def create_dataclass_with_a_b_c():
    return ComplexDataClass(1, 2, 3)


def create_dataclass_with_kwarg_a_b_c():
    return ComplexDataClass(1, 2, 3)


class TestDataClassConstruction(TestCaseBase):
    def test_create_dataclass_with_a(self):
        self.assert_results(create_dataclass_with_a)

    def test_create_dataclass_with_kwarg_a(self):
        self.assert_results(create_dataclass_with_kwarg_a)

    def test_create_dataclass_with_a_b_c(self):
        self.assert_results(create_dataclass_with_a_b_c)

    def test_create_dataclass_with_kwarg_a_b_c(self):
        self.assert_results(create_dataclass_with_kwarg_a_b_c)


if __name__ == "__main__":
    unittest.main()
