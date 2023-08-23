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

from cinn import ir


class ComputeCodeGenerator(ast.NodeVisitor):
    """
    Convert python ast to CINN Lower Level IR,
    containing only the semantics of the compute part
    """

    def __init__(self, function_name, inputs_signature):
        self.function_name = function_name
        self.inputs_signature = inputs_signature
        self.cinn_llir_func = None
        self.left_value_scope = {}
        self.local_variables = {}

    def visit_FunctionDef(self, node) -> None:
        """
        Parse CINN Low Level IR FunctionDef.

        Args:
            node(ast.FunctionDef): The ast FunctionDef Node
        """
        arg_names = self.visit(node.args)

        assert len(node.args.defaults) == 0, "Not support default args"

        # 1. Construct args of function
        llir_args = []
        for i, arg_name in enumerate(arg_names):
            if hasattr(self.inputs_signature[i], "dtype"):
                llir_value = ir._Buffer_.make(
                    "_" + arg_name, self.inputs_signature[i].dtype
                )
                llir_args.append(
                    ir.Argument(llir_value, ir.Argument.IO.kUnknown)
                )
                tensor_shape = [
                    ir.Expr(dim) for dim in self.inputs_signature[i].shape
                ]
                llir_value = ir._Tensor_.make(
                    arg_name,
                    self.inputs_signature[i].dtype,
                    tensor_shape,
                    tensor_shape,
                )
            else:
                llir_value = ir.Var(arg_name)
                llir_args.append(ir.Argument(llir_value))
                llir_value = ir.Expr(llir_value)
            self.set_value(arg_name, llir_value)

        # 2. Construct body of function
        stmts = self.visit_compound_statement(node.body)
        body = ir.Block.make(stmts)

        # 3. Construct LoweredFunc
        self.cinn_llir_func = ir.LoweredFunc.make(
            self.function_name, llir_args, body
        )

    def visit_compound_statement(self, stmts):
        cinn_stmts = []
        for stmt in stmts:
            cinn_stmt = self.visit(stmt)
            assert cinn_stmt is not None, f"Unsupport parse:\n{stmt}"
            if cinn_stmt is None:
                continue
            cinn_stmts.append(self.visit(stmt))
        return cinn_stmts

    def visit_arguments(self, node):
        arg_names = []
        for arg in node.args:
            arg_names += [self.visit(arg)]
        return arg_names

    def visit_arg(self, node):
        ast.NodeVisitor.generic_visit(self, node)
        return node.arg

    def visit_For(self, node) -> ir.Expr:
        """
        parse CINN Low Level IR For.

        Args:
            node(ast.For): The ast For node

        Returns:
            ir.Expr, Points to the Expr of ir::ExprNode<For>
        """
        iter_args = [self.visit(arg) for arg in node.iter.args]
        assert (
            len(iter_args) <= 2
        ), "CINN Low Level IR does not support setting the range step"
        ast_min = iter_args[0] if len(iter_args) > 1 else self.visit(ast.Num(0))
        ast_extent = iter_args[1] if len(iter_args) > 1 else iter_args[0]

        # TODO(6clc): support sub region's local variable
        llir_var = ir.Var(node.target.id)
        llir_var_expr = ir.Expr(llir_var)
        self.set_value(node.target.id, llir_var_expr)

        llir_for_min = ir.Expr(ast_min)
        llir_for_extent = ir.Expr(ast_extent)
        llir_for_body = self.visit_compound_statement(node.body)
        llir_for_body = ir.Block.make(llir_for_body)
        for_expr = ir.For.make(
            llir_var, llir_for_min, llir_for_extent, llir_for_body
        )
        return for_expr

    def visit_Name(self, node):
        if type(node.ctx) == ast.Store:
            if node.id in self.local_variables:
                return self.local_variables[node.id]
            return node.id
        # Load Node
        assert (
            node.id in self.local_variables
        ), f"{node.id} is not defined in context"
        return self.local_variables[node.id]

    def visit_BinOp(self, node):
        cinn_tensor_l, indexs_l = self.visit(node.left)
        lhs = ir.Load.make(cinn_tensor_l, indexs_l)
        cinn_tensor_r, indexs_r = self.visit(node.right)
        rhs = ir.Load.make(cinn_tensor_r, indexs_r)
        ast2cinn = {ast.Add: ir.Add}
        return ast2cinn[ast.Add].make(lhs, rhs)

    def visit_Subscript(self, node):
        lhs_tensor = self.visit(node.value)
        idxs = [
            self.visit(node.slice),
        ]
        return lhs_tensor.Expr(), idxs

    def visit_Tuple(self, node):
        args = [self.visit(x) for x in node.elts]
        return args

    def visit_Index(self, node):
        return self.visit(node.value)

    def visit_Assign(self, node):
        """
        parse CINN Low Level IR Store.

        Args:
            node(ast.Assign): The ast Assign node

        Returns:
            ir.Expr, Points to the Expr of ir::ExprNode<Store>
        """

        _names = []
        for target in node.targets:
            _names += [self.visit(target)]
        assert (
            len(_names) == 1
        ), "Unsupport targets is a \
               list of nodes, like 'a, b = c'"
        names = _names[0]
        value = self.visit(node.value)

        return ir.Store.make(names[0], value, names[1])

    def set_value(self, name, value):
        self.left_value_scope[name] = value
        self.local_variables[name] = value

    def visit_Constant(self, node):
        return ir.Expr(node.value)
