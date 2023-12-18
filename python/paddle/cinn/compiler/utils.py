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

try:
    from _collections import defaultdict
except ImportError:
    pass


from paddle.cinn.schedule import IRSchedule


def is_node_parsed_in_schedule(node: ast.Call):
    func_name = ""
    if isinstance(node.func, ast.Name):
        func_name = node.func.id
    elif isinstance(node.func, ast.Attribute):
        func_name = node.func.attr
    if func_name == "make":
        return False
    if func_name == "print":
        return True

    return getattr(IRSchedule, func_name, None)


def node_is_schedule_block_context(node: ast.Call):
    if isinstance(node.func, ast.Name):
        return node.Name == "ScheduleBlockContext"
    if isinstance(node.func, ast.Attribute):
        return node.func.attr == "ScheduleBlockContext"
    return False


class VariableTable:
    def __init__(self):
        # var name added by current context
        self.var_name_list = []
        # var name to var. Dtype is {string:list}
        # list records the value assigned to each layer of context
        self.name2value = defaultdict(list)

    def __enter__(self):
        self.var_name_list.append([])
        return self

    def __exit__(self, ptype, value, trace) -> None:
        # clear var assign in current context
        if ptype is None and value is None:
            var_names = self.var_name_list.pop()
            for var_name in var_names:
                self.name2value[var_name].pop()
                if len(self.name2value[var_name]) == 0:
                    self.name2value.pop(var_name)

    def add(self, name, value, cover=False):
        if cover and name in self.var_name_list[-1]:
            self.name2value[name][-1] = value
        else:
            self.var_name_list[-1].append(name)
            self.name2value[name].append(value)

    def get(self):
        return {k: v[-1] for k, v in self.name2value.items()}
