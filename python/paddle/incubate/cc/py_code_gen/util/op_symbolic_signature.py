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
from typing import TYPE_CHECKING

from paddle.incubate.cc.py_code_gen.util.hash_combine import hash_combine

if TYPE_CHECKING:
    from paddle.incubate.cc.py_code_gen.ir.ir_symbol import ShapeOrDataDimExprs
    from paddle.incubate.cc.py_code_gen.util.op_stringized_expr import (
        OpStringizedExpr,
    )


@dataclass
class OpSymbolicSignature:
    op_expr: OpStringizedExpr
    inputs_dim_exprs: list[ShapeOrDataDimExprs]
    outputs_dim_exprs: list[ShapeOrDataDimExprs]

    def __hash__(self):
        hash_value = hash(self.op_expr)
        for symbolic_dim_exprs in self.all_dim_exprs():
            hash_value = hash_combine(hash_value, hash(symbolic_dim_exprs))
        return hash_value

    def all_dim_exprs(self):
        yield from self.inputs_dim_exprs
        yield from self.outputs_dim_exprs
