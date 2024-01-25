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

from paddle.cinn.schedule import IRSchedule

from .expr_executor import ExprExecutor, exec_assign
from .utils import (
    VariableTable,
    is_node_parsed_in_schedule,
    node_is_schedule_block_context,
)


class ScheduleCodeGenerator(ast.NodeVisitor):
    """
    Convert python ast to CINN Lower Level IR,
    containing only the semantics of the schedule part
    """

    def __init__(self, fn, cinn_llir_func):
        self.fn = fn
        self.cinn_llir_func = cinn_llir_func
        self.scheduler = IRSchedule.make(self.cinn_llir_func)
        self.variable_table = VariableTable()
        self.global_variable_table = VariableTable()
        # Set the schedule-related variable to global
        self.extra_scope = {
            "ScheduleBlockVariable": ScheduleBlockVariable,
            "scheduler": self.scheduler,
        }
        self.loop_var_stack = []
        self.block_stack = []
        self.sch_block_tmp_var_name = "__CINN_SCHEDULE_BLOCK_VAR_NAME__"
        self.tmp_var_count = 1

    def parse(self):
        with self.variable_table, self.global_variable_table:
            ast_node = self.fn.parse()
            for k, v in self.fn.scope.items():
                self.variable_table.add(k, v)
            for k, v in self.extra_scope.items():
                self.variable_table.add(k, v)
            self.visit(ast_node)
        return self.cinn_llir_func

    def visit_For(self, node):
        assert isinstance(
            node.target, ast.Name
        ), "Current only support range() to make ForLoop"
        with self.variable_table:
            self.loop_var_stack.append(node.target)
            self.generic_visit(node)
            self.loop_var_stack.pop()

    def visit_compound_statement(self, stmts):
        for stmt in stmts:
            self.visit(stmt)

    def visit_With(self, node):
        with self.variable_table:
            for item in node.items:
                if isinstance(
                    item.context_expr, ast.Call
                ) and not node_is_schedule_block_context(item.context_expr):
                    continue
                # 1. replace ScheduleBlockContext to ScheduleBlockVariable
                sch_ctx_node = item.context_expr
                sch_block_node = ast.copy_location(
                    ast.Call(
                        func=ast.Name(
                            id="ScheduleBlockVariable", ctx=ast.Load()
                        ),
                        args=sch_ctx_node.args,
                        keywords=[],
                        starargs=None,
                        kwargs=None,
                    ),
                    item.context_expr,
                )
                item.context_expr = sch_block_node

                # 2. store ScheduleBlockVariable node
                sch_block = ExprExecutor(self.variable_table.get()).exec(
                    item.context_expr
                )
                if item.optional_vars is None:
                    tmp_var_name = self.sch_block_tmp_var_name + str(
                        self.tmp_var_count
                    )
                    sch_block_var_node = ast.Name(
                        id=tmp_var_name, ctx=ast.Store()
                    )
                    item.optional_vars = sch_block_var_node
                local_var_table = exec_assign(
                    target=item.optional_vars, source=sch_block
                )
                # 3. Set the block's loop to its attribute
                sch_block.set_scheduler(self.scheduler)
                self.block_stack.append(sch_block)
                for k, v in local_var_table.items():
                    self.variable_table.add(k, v)
                    self.global_variable_table.add(k, v)
                    for loop_var in self.loop_var_stack:
                        loop_var_value = ast.Attribute(
                            value=ast.Name(id=k, ctx=ast.Load()),
                            attr=loop_var.id,
                            ctx=ast.Load(),
                        )
                        loop_var_value = ExprExecutor(
                            self.variable_table.get()
                        ).exec(loop_var_value)
                        for_loop_var_table = exec_assign(
                            loop_var, loop_var_value
                        )
                        for (
                            loop_var_k,
                            loop_var_v,
                        ) in for_loop_var_table.items():
                            self.variable_table.add(loop_var_k, loop_var_v)

            body = self.visit_compound_statement(node.body)

    def visit_Assign(self, node):
        if isinstance(node.value, ast.Call) and is_node_parsed_in_schedule(
            node.value
        ):
            sch_ret = self.exec_schedule_primitive(node.value)
            local_var_table = exec_assign(
                target=node.targets[0], source=sch_ret
            )
            for k, v in local_var_table.items():
                self.variable_table.add(k, v)
            return
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node, ast.Call) and is_node_parsed_in_schedule(node):
            self.exec_schedule_primitive(node)
            return

    def exec_schedule_primitive(self, node):
        # Reflect ScheduleBlockContext to ScheduleBlockVariable
        sch_primitive = node
        args = [ast.Name(id="scheduler", ctx=ast.Load()), *sch_primitive.args]
        sch_primitive.args = args
        all_variable_table = self.variable_table.get()
        for k, v in self.global_variable_table.get().items():
            all_variable_table[k] = v
        sch_ret = ExprExecutor(all_variable_table).exec(node)

        return sch_ret


class ScheduleBlockVariable:
    """
    The parse Schedule process replaces ScheduleBlockContext with this class on the ast layer to improve schedule usability on the python layer
    For example, split a loop in c++ requires two steps:
        1. Gets the loop for the corresponding block: `x, y = sch.get_loops(block)`
        2. Apply schedule to loop: tx, xi = sch.split(x, [2])
    This class allows you to directly manipulate the loop name of a block
        `sch.split(block.x, [2])`
    """

    def __init__(self, name):
        self.name = name
        self.scheduler = None

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def __getattr__(self, k):
        if k == "block":
            return self.scheduler.get_block(self.name)
        else:
            name2loops = self.scheduler.get_name2loops_dict(self.name)
            return name2loops[k]
