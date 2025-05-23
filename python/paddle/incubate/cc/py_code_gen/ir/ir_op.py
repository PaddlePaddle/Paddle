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

from dataclasses import dataclass

from paddle.incubate.cc.py_code_gen.ir import ir_tensor


@dataclass
class Op:
    name: str
    op_id: int
    input_types: list
    output_types: list
    attrs: dict
    block_positional_arg_names: list[list[list[str]]] | None
    block_keyword_arg_names: list[list[list[str]]] | None
    block_positional_arg_types: list[list[list[Type]]] | None  # noqa: F821
    block_keyword_arg_types: list[list[list[Type]]] | None  # noqa: F821
    __operands_symbols_signature__: ArrayAttribute = None  # noqa: F821
    __results_symbols_signature__: ArrayAttribute = None  # noqa: F821

    def GetResults(self):
        return [self.GetResult(i) for i in range(len(self.output_types))]

    def GetResult(self, i):
        return ir_tensor.Tensor(
            local_name_prefix=self.GetNameSuffix(),
            name=self.GetResultTensorName(i),
            arg_name_as_input=self.GetArgNameAsInput(),
            defining_op_name=self.name,
            type=self.output_types[i],
            dim_exprs=self.__results_symbols_signature__.value[i].value,
        )

    def GetArgNameAsInput(self):
        if self.name == "pd_op.data":
            return self.attrs["name"].value
        if self.name == "pd_op.feed":
            return self.attrs["name"].value
        if self.name == "builtin.parameter":
            return self.attrs["parameter_name"].value
        if self.name == "builtin.constant":
            return self.attrs["value"].value
        return None

    def GetResultTensorName(self, i):
        return f"{self.GetUniqueName()}{i}"

    def GetUniqueName(self):
        return f"{self.GetNameSuffix()}_{self.op_id}"

    def GetPyVarName(self):
        return "_".join(self.GetValidPyVarNameComponents())

    def GetNameSuffix(self):
        return self.GetValidPyVarNameComponents()[-1]

    def GetValidPyVarNameComponents(self):
        def IsValidVarChar(ch):
            return (
                (ch >= "a" and ch <= "z")
                or (ch >= "A" and ch <= "Z")
                or (ch >= "0" and ch <= "9")
                or ch == "_"
            )

        ret = ""
        for i in range(len(self.name)):
            ret += self.name[i] if IsValidVarChar(self.name[i]) else ":"
        return ret.split(":")
