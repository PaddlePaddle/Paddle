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

from paddle.incubate.cc.py_code_gen.ir import ir_symbol


class DimExprEvaluator:

    def __init__(self, get_symbol_binding):
        self.get_symbol_binding = get_symbol_binding

    def Eval(self, dim_expr):
        return getattr(self, f"{type(dim_expr).__name__}")(dim_expr)

    def Int64(self, dim_expr):
        return int(dim_expr.value)

    def String(self, dim_expr):
        return int(self.get_symbol_binding(dim_expr.value))

    def Negative(self, dim_expr):
        return int(-self.Eval(dim_expr.operand))

    def Reciprocal(self, dim_expr):
        raise NotImplementedError("Invalid DimExpr")

    def Add(self, dim_expr):
        dim_instance = 1
        for operand in dim_expr.operands:
            if isinstance(operand, ir_symbol.Negative):
                dim_instance -= self.Eval(operand.operand)
            else:
                dim_instance += self.Eval(operand)
        return int(dim_instance)

    def Mul(self, dim_expr):
        dim_instance = 1
        for operand in dim_expr.operands:
            if isinstance(operand, ir_symbol.Reciprocal):
                dim_instance //= self.Eval(operand.operand)
            else:
                dim_instance *= self.Eval(operand)
        return int(dim_instance)

    def Max(self, dim_expr):
        return max(*[self.Eval(operand) for operand in dim_expr.operands])

    def Min(self, dim_expr):
        return min(*[self.Eval(operand) for operand in dim_expr.operands])

    def Broadcast(self, dim_expr):
        return max(*[self.Eval(operand) for operand in dim_expr.operands])
