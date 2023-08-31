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

    def visit_Subscript(self, node):
        """
        save block information
        """
        if type(node.ctx) != ast.Store:
            return
        block = self.scheduler.get_block(node.value.id)
        self.name2blocks[node.value.id] = block
        loops = self.scheduler.get_loops(node.value.id)
        name2loops = self.scheduler.get_name2loops_dict(node.value.id)

        for sch_node in self.sch_seq:
            sch_name = (
                sch_node.func.id
                if isinstance(sch_node.func, ast.Name)
                else sch_node.func.attr
            )
            sch_args = []
            for sch_arg in sch_node.args:
                if isinstance(sch_arg, ast.Name):
                    assert (
                        sch_arg.id in name2loops
                        or sch_arg.id in self.name2blocks
                    ), f'No matching block and loop was found for {sch_name}, \
                    current blocks are {self.name2blocks.keys()}, loops under block {node.value.id} are {name2loops.keys()}'
                    sch_args.append(name2loops[sch_arg.id])
                else:
                    sch_args.append(sch_arg)

            # TODO(6clc): support keywords
            # sch_keywords = sch_node.keywords
            getattr(self.scheduler, sch_name)(sch_args)

        self.sch_seq = []

    def visit_Assign(self, node):
        if not isinstance(node.value, ast.Call):
            self.generic_visit(node)
            return
        if isinstance(node.value.func, ast.Name) and node.func.id not in [
            "fuse",
            "split",
        ]:
            self.sch_seq.append(node)
            return
        if isinstance(
            node.value.func, ast.Attribute
        ) and node.func.attr not in [
            "fuse",
            "split",
        ]:
            self.sch_seq.append(node)
            return
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id not in [
            "fuse",
            "split",
        ]:
            return
        if isinstance(node.func, ast.Attribute) and node.func.attr not in [
            "fuse",
            "split",
        ]:
            return
        self.sch_seq.append(node)
