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

from cinn.schedule import IRSchedule

from .utils import node_is_schedule


class ScheduleCodeGenerator(ast.NodeVisitor):
    """
    Convert python ast to CINN Lower Level IR,
    containing only the semantics of the schedule part
    """

    def __init__(self, cinn_llir_func):
        self.cinn_llir_func = cinn_llir_func
        self.scheduler = IRSchedule.make(self.cinn_llir_func)
        self.sch_seq = []
        self.name2blocks = {}
        self.name2loops = {}

    def visit_Subscript(self, node):
        """
        save block information
        """
        if type(node.ctx) != ast.Store:
            return

        for sch_node in self.sch_seq:
            self.name2blocks[node.value.id] = self.scheduler.get_block(
                node.value.id
            )
            self.name2loops = self.scheduler.get_name2loops_dict(node.value.id)

            sch_name = (
                sch_node.func.id
                if isinstance(sch_node.func, ast.Name)
                else sch_node.func.attr
            )
            sch_args = [self.eval(item) for item in sch_node.args]

            sch_keywords = {
                kw.arg: self.eval(kw.value) for kw in sch_node.keywords
            }

            getattr(self.scheduler, sch_name)(*sch_args, **sch_keywords)

        self.sch_seq = []
        self.name2loops = {}

    def visit_Assign(self, node):
        if isinstance(node.value, ast.Call) and node_is_schedule(node):
            self.sch_seq.append(node)
            return
        self.generic_visit(node)

    def visit_Call(self, node):
        if not node_is_schedule(node):
            return
        self.sch_seq.append(node)

    def eval(self, node):
        return getattr(self, f'eval_{type(node).__name__}')(node)

    def eval_List(self, node):
        return [self.eval(item) for item in node.elts]

    def eval_Tuple(self, node):
        return [self.eval(item) for item in node.elts]

    def eval_Constant(self, node):
        return node.value

    def eval_UnaryOp(self, node):
        return eval(
            compile(ast.Expression(body=node), filename='', mode='eval')
        )

    def eval_Name(self, node):
        if node.id in self.name2loops:
            return self.name2loops[node.id]
        elif node.id in self.name2blocks:
            return self.name2blocks[node.id]
        else:
            raise Exception(
                f'No matching block and loop was found for {node.id}. \
                 Current loops are {self.name2loops.keys()}. \
                 Current blocks are {self.name2blocks.keys()}.'
            )
