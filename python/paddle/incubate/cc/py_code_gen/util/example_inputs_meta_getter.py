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
from functools import reduce

from paddle.incubate.cc.py_code_gen.ir import ir_type

InputMetaKey = namedtuple("InputMetaKey", ["program_id", "input_name"])


@dataclass
class InputMeta:
    name: str
    shape: list[int]
    data: list[int] | None


class ExampleInputsMetaGetter:
    def __init__(self, records):
        self.input_meta_key2value = self._MakeInputMetaKey2Value(records)

    def HasAllInputExamples(self, program_id, input_tensors) -> bool:
        return all(
            self.Has(program_id, input_tensor) for input_tensor in input_tensors
        )

    def Has(self, program_id, input_tensor) -> InputMeta:
        key = InputMetaKey(program_id, input_tensor.arg_name_as_input)
        if key in self.input_meta_key2value:
            return True
        static_shape = self.GetInputStaticShape(input_tensor)
        if static_shape is None:
            return False
        if input_tensor.defining_op_name == "builtin.parameter":
            return True
        if input_tensor.defining_op_name == "builtin.constant":
            return True
        if self.IsSmallIntegerTensor(static_shape, input_tensor.dtype):
            return False
        return True

    def IsSmallIntegerTensor(self, shape, dtype):
        return self.IsSmallTensor(shape) and self.IsIntegerTensor(dtype)

    def IsSmallTensor(self, shape):
        if len(shape) == 0:
            return True
        return reduce(lambda x, y: x * y, shape) <= 64

    def IsIntegerTensor(self, dtype):
        return isinstance(dtype, (ir_type.Int32Type, ir_type.Int64Type))

    def Get(self, program_id, input_tensor) -> InputMeta:
        key = InputMetaKey(program_id, input_tensor.arg_name_as_input)
        if key in self.input_meta_key2value:
            return self.input_meta_key2value[key]
        shape = self.GetInputStaticShape(input_tensor)
        if shape is None:
            raise NotImplementedError
        return InputMeta(
            name=input_tensor.arg_name_as_input, shape=shape, data=None
        )

    def GetInputStaticShape(self, input_tensor):
        if not isinstance(input_tensor.type, ir_type.DenseTensorType):
            return None
        for dim in input_tensor.shape:
            if dim < 0:
                return None
        return input_tensor.shape

    def _MakeInputMetaKey2Value(self, records):
        input_meta_key2value = {}
        for record in records:
            key = InputMetaKey(record.program_id, record.input_name)
            input_meta_key2value[key] = InputMeta(
                name=record.input_name,
                shape=record.shape,
                data=record.data if hasattr(record, "data") else None,
            )
        return input_meta_key2value
