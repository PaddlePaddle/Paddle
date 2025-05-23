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
from paddle.incubate.cc.py_code_gen.util.dim_expr_evaluator import (
    DimExprEvaluator,
)
from paddle.incubate.cc.py_code_gen.util.global_dim_expr_converter import (
    GlobalDimExprConverter,
)


class DimInstanceGenerator:

    def __init__(self, dim_exprs):
        self.global_dim_expr_converter = GlobalDimExprConverter(dim_exprs)
        local_dim_exprs = [
            self.global_dim_expr_converter.GetLocalDimExpr(dim_expr)
            for dim_expr in dim_exprs
        ]
        symbol_binding_value = self.SearchValidSymbolBindingValue(
            local_dim_exprs
        )
        self.dim_expr_evaluator = DimExprEvaluator(
            lambda name: symbol_binding_value
        )
        self.example_dim = -1

    def GetDimInstance(self, dim_expr):
        local_dim_expr = self.global_dim_expr_converter.GetLocalDimExpr(
            dim_expr
        )
        dim_instance = self.dim_expr_evaluator.Eval(local_dim_expr)
        return dim_instance if dim_instance > 0 else self.example_dim

    def SearchValidSymbolBindingValue(self, dim_exprs):
        threshold = 0
        try_cnt = 64
        for i in range(2, try_cnt):
            dim_expr_evaluator = DimExprEvaluator(lambda name: i)
            if self.AllDimExprGreaterThan(
                threshold, dim_exprs, dim_expr_evaluator
            ):
                return i
        return 0

    def AllDimExprGreaterThan(self, threshold, dim_exprs, dim_expr_evaluator):
        def Ignored(dim_expr):
            if isinstance(dim_expr, ir_symbol.Int64):
                return True
            if isinstance(dim_expr, ir_symbol.Negative):
                return isinstance(dim_expr.operand, ir_symbol.Int64)
            if isinstance(dim_expr, ir_symbol.Reciprocal):
                return isinstance(dim_expr.operand, ir_symbol.Int64)

        for dim_expr in dim_exprs:
            if Ignored(dim_expr):
                continue
            if dim_expr_evaluator.Eval(dim_expr) < threshold:
                return False
        return True
