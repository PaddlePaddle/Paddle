# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import ast

from paddle.cinn import ir

# The Python native AST node that cinn ir supports
AST2CINN = {
    ast.Add: ir.Add,
    ast.Sub: ir.Sub,
    ast.Mult: ir.Mul,
    ast.Div: ir.Div,
    ast.Mod: ir.Mod,
    ast.And: ir.And,
    ast.Or: ir.Or,
    ast.USub: ir.Minus,
    ast.Not: ir.Not,
    ast.Eq: ir.EQ,
    ast.NotEq: ir.NE,
    ast.Lt: ir.LT,
    ast.LtE: ir.LE,
    ast.Gt: ir.GT,
    ast.GtE: ir.GE,
}


class ExprExecutor:
    def __init__(self, var_table):
        self.var_table = var_table
        self.tmp_value_count = 1

    def exec(self, node):
        ret = self.visit(node)
        if isinstance(ret, ast.Name):
            return self.var_table[ret.id]
        if isinstance(ret, ast.Constant):
            return ret.value
        raise Exception(f"Error result type: {type(ret)}")

    def visit(self, node):
        if isinstance(node, list):
            return [self.visit(item) for item in node]
        if isinstance(node, tuple):
            return (self.visit(item) for item in node)
        assert isinstance(node, ast.AST)
        if isinstance(node, ast.Name):
            return node

        if isinstance(node, ast.Constant):
            return node

        if not isinstance(node, (ast.expr, ast.slice)):
            # some nodes don't need to parse, such as ast.Load
            return node
        if isinstance(node, (ast.Lambda, ast.Starred)):
            raise Exception("Current not supported: Lambda, Starred")

        cls_fields = {}
        for field in node.__class__._fields:
            attr = getattr(node, field)
            if isinstance(attr, (ast.AST, tuple, list)):
                cls_fields[field] = self.visit(attr)
            else:
                cls_fields[field] = attr

        node_type_name = f'eval_{type(node).__name__}'
        if hasattr(self, node_type_name):
            exec_func = getattr(self, node_type_name)
            value = exec_func(cls_fields)
        else:
            new_node = node.__class__(**cls_fields)
            value = self.exec_expr(new_node)
        return self.save_temp_value(value)

    def exec_expr(self, node):
        assert isinstance(node, ast.expr)
        if type(node).__name__ == "Constant":
            return node.value

        node = ast.Expression(node)
        node = ast.fix_missing_locations(node)
        exec = compile(node, filename="<ast>", mode="eval")
        return eval(exec, self.var_table)

    def eval_BinOp(self, fields):
        args = [self.exec_expr(fields["left"]), self.exec_expr(fields["right"])]
        args = [
            ir.Expr(item) if not isinstance(item, ir.Expr) else item
            for item in args
        ]
        return AST2CINN[type(fields["op"])].make(*args)

    def eval_UnaryOp(self, fields):
        args = [self.exec_expr(fields["operand"])]
        args = [
            ir.Expr(item) if not isinstance(item, ir.Expr) else item
            for item in args
        ]
        return AST2CINN[type(fields["op"])].make(*args)

    def eval_Compare(self, fields):
        assert (
            len(fields["ops"]) == 1
        ), "Only binary comparison symbols are supported. Expressions such as '1 <= a < 10' are not supported."
        args = [
            self.exec_expr(fields["left"]),
            self.exec_expr(fields["comparators"][0]),
        ]
        args = [
            ir.Expr(item) if not isinstance(item, ir.Expr) else item
            for item in args
        ]
        return AST2CINN[type(fields["ops"][0])].make(*args)

    def save_temp_value(self, value):
        name = f"__cinn_python_script_tmp_value_{self.tmp_value_count}"
        self.tmp_value_count += 1
        self.var_table[name] = value
        return ast.Name(
            id=name,
            ctx=ast.Load(
                lineno=0, col_offset=0, end_lineno=None, end_col_offset=None
            ),
            lineno=0,
            col_offset=0,
            end_lineno=None,
            end_col_offset=None,
        )


def exec_assign(target, source):
    right_value_var_name = "__CINN_RIGHT_VALUE_VAR_NAME__"
    local_var_table = {right_value_var_name: source}
    mod = ast.fix_missing_locations(
        ast.Module(
            body=[
                ast.Assign(
                    targets=[target],
                    value=ast.Name(id=right_value_var_name, ctx=ast.Load()),
                )
            ],
            type_ignores=[],
        )
    )
    exe = compile(mod, filename="<ast>", mode="exec")
    exec(exe, {}, local_var_table)
    del local_var_table[right_value_var_name]
    return local_var_table
