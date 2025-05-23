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

from collections import namedtuple
from dataclasses import dataclass

from paddle.incubate.cc.py_code_gen.ir import ir_type

OpInputMetaKey = namedtuple(
    "InputMetaKey", ["program_id", "op_id", "input_idx"]
)

ShapeType = list[int]


@dataclass
class OpInputMeta:
    op_id: int
    input_idx: int
    shape: ShapeType | list[ShapeType] | None
    data: ShapeType | list[ShapeType] | None


class OpExampleInputsMetaGetter:
    def __init__(self, records):
        self.input_meta_key2value = self._MakeOpInputMetaKey2Value(records)

    def HasAllInputs(self, program_id, op) -> bool:
        op_id = op.op_id
        num_inputs = len(op.input_types)
        for input_idx in range(num_inputs):
            if isinstance(op.input_types[input_idx], ir_type.NullType):
                continue
            key = OpInputMetaKey(program_id, op_id, input_idx)
            if key not in self.input_meta_key2value:
                return False
        return True

    def Get(self, program_id, op_id, input_idx) -> OpInputMeta:
        key = OpInputMetaKey(program_id, op_id, input_idx)
        return self.input_meta_key2value.get(key, None)

    def _MakeOpInputMetaKey2Value(self, records):
        input_meta_key2value = {}
        for record in records:
            key = OpInputMetaKey(
                record.program_id,
                record.op_id,
                record.input_idx,
            )
            input_meta_key2value[key] = OpInputMeta(
                op_id=record.op_id,
                input_idx=record.input_idx,
                shape=record.shape,
                data=record.data,
            )
        return input_meta_key2value


def MakeOpExampleInputsMetaGetter(name_and_classes):
    classes = [
        cls
        for name, cls in name_and_classes
        if name.startswith("PirProgram_op_input_tensor_meta_")
    ]
    return OpExampleInputsMetaGetter(records=classes)
