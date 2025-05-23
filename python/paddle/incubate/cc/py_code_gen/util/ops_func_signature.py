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

import typing as t
from collections import namedtuple
from dataclasses import dataclass
from typing import TYPE_CHECKING

from paddle.incubate.cc.py_code_gen.ir import ir_type

if TYPE_CHECKING:
    from paddle.incubate.cc.py_code_gen.util.input_tensor_desc import (
        InputTensorDesc,
    )

InputSpecDesc = namedtuple(
    "InputSpecDesc",
    [
        "shape",
        "dtype",
    ],
)


@dataclass
class TensorId:
    pass


@dataclass
class NullTensorId(TensorId):
    pass

    def get_source_name(
        self, get_source_names: t.Callable[int, list[str]]
    ) -> str:
        return "None"


@dataclass
class OperandTensorId(TensorId):
    op_id: int
    operand_tensor_idx: int

    def get_source_name(
        self, get_source_names: t.Callable[int, list[str]]
    ) -> str:
        return get_source_names(self.op_id)[self.operand_tensor_idx]


@dataclass
class TensorListMemberId(TensorId):
    op_id: int
    operand_tensor_list_idx: int
    tensor_list_member_idx: int

    def get_source_name(
        self, get_source_names: t.Callable[int, list[str]]
    ) -> str:
        source_name = get_source_names(self.op_id)[self.operand_tensor_list_idx]
        return f"{source_name}_{self.tensor_list_member_idx}"


@dataclass
class OperandId:
    op_id: int
    operand_idx: int

    def get_operand_tensor_id(self, op) -> OperandTensorId | None:
        input_type = op.input_types[self.operand_idx]
        if not isinstance(input_type, ir_type.DenseTensorType):
            return None
        return OperandTensorId(self.op_id, self.operand_idx)

    def get_null_tensor_id(self, op) -> NullTensorId | None:
        input_type = op.input_types[self.operand_idx]
        if not isinstance(input_type, ir_type.NullType):
            return None
        return NullTensorId()

    def get_tensor_list_member_ids(self, op) -> list[TensorListMemberId] | None:
        input_type = op.input_types[self.operand_idx]
        if not isinstance(input_type, ir_type.VectorType):
            return None
        return [
            TensorListMemberId(
                self.op_id,
                operand_tensor_list_idx=self.operand_idx,
                tensor_list_member_idx=i,
            )
            for i in range(len(input_type.value))
        ]

    def get_source_name(
        self, get_source_names: t.Callable[int, list[str]]
    ) -> str:
        return get_source_names(self.op_id)[self.operand_idx]


@dataclass
class OpsFuncSignature:
    tensor_ids: list[TensorId]
    operand_ids: list[OperandId]
    operand_tensor_id4operand_id: t.Callable[OperandId, OperandTensorId | None]
    null_tensor_id4operand_id: t.Callable[OperandId, NullTensorId | None]
    tensor_list_member_ids4operand_id: t.Callable[
        OperandId, list[TensorListMemberId] | None
    ]
    tensor_name4tensor_id: t.Callable[TensorId, str]
    tensor_name4operand_id: t.Callable[OperandId, str]
    input_spec_shape_dtype4tensor_id: t.Callable[TensorId, InputSpecDesc]
    example_input_meta4tensor_id: t.Callable[TensorId, InputTensorDesc]
    example_input_data4operand_id: t.Callable[OperandId, list[int] | None]
    immediate_value4operand_id: t.Callable[[OperandId, t.Any], t.Any]
    immediate_value4int_array_member_id: t.Callable[[TensorId, t.Any], t.Any]
