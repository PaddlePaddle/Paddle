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

if TYPE_CHECKING:
    from paddle.incubate.cc.py_code_gen.ir import ir_symbol
from paddle.incubate.cc.py_code_gen.util.hash_combine import hash_combine


@dataclass
class ConstraintRecord:
    pass


@dataclass
class EqualConstraintRecord(ConstraintRecord):
    lhs: ir_symbol.DimExpr
    rhs: ir_symbol.DimExpr


@dataclass
class BroadcastableConstraintRecord(ConstraintRecord):
    lhs: ir_symbol.DimExpr
    rhs: ir_symbol.DimExpr


@dataclass
class GtOneConstraintRecord(ConstraintRecord):
    value: ir_symbol.DimExpr


@dataclass
class Constraint:
    pass


@dataclass
class NoConstraint(Constraint):
    no_dim_exprs: list[ir_symbol.DimExpr]

    def __hash__(self):
        hash_value = id(NoConstraint)
        for dim_expr in self.equal_dim_exprs:
            hash_value = hash_combine(hash_value, hash(dim_expr))
        return hash_value


@dataclass
class EqualConstraint(Constraint):
    equal_dim_exprs: list[ir_symbol.DimExpr]

    def __hash__(self):
        hash_value = id(EqualConstraint)
        for dim_expr in self.equal_dim_exprs:
            hash_value = hash_combine(hash_value, hash(dim_expr))
        return hash_value


@dataclass
class BroadcastableConstraint(Constraint):
    braodcastable_dim_exprs: list[ir_symbol.DimExpr]

    def __hash__(self):
        hash_value = id(BroadcastableConstraint)
        for dim_expr in self.braodcastable_dim_exprs:
            hash_value = hash_combine(hash_value, hash(dim_expr))
        return hash_value


@dataclass
class GtOneConstraint(Constraint):
    gt_one_dim_expr: ir_symbol.DimExpr

    def __hash__(self):
        hash_value = id(GtOneConstraint)
        hash_value = hash_combine(hash_value, hash(self.gt_one_dim_expr))
        return hash_value


@dataclass
class SymmetricDimVar:
    pass


@dataclass
class SymbolSymmetricDimVar(SymmetricDimVar):
    symbol: str


@dataclass
class ComposedSymmetricDimVar(SymmetricDimVar):
    symmetric_dim_vars: list[SymmetricDimVar]


@dataclass
class AnySymmetricDimVar(ComposedSymmetricDimVar):
    pass


@dataclass
class AddSymmetricDimVar(ComposedSymmetricDimVar):
    pass


@dataclass
class MulSymmetricDimVar(ComposedSymmetricDimVar):
    pass


@dataclass
class MaxSymmetricDimVar(ComposedSymmetricDimVar):
    pass


@dataclass
class MinSymmetricDimVar(ComposedSymmetricDimVar):
    pass


@dataclass
class BroadcastSymmetricDimVar(ComposedSymmetricDimVar):
    pass
