# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from collections import OrderedDict
from typing import List

from paddle.fluid import core
from paddle.fluid.framework import Program


def list_to_ordered_dict(list_obj, ordered_dict=None):
    if ordered_dict is None:
        ordered_dict = OrderedDict()
    else:
        assert isinstance(ordered_dict, OrderedDict)
    for obj in list_obj:
        if obj not in ordered_dict:
            ordered_dict[obj] = True
    return ordered_dict


# The inputs of a program are the variables
# that first occur as the input of the op.
def get_inputs_of_program(program):
    visited_vars = set()
    input_vars = []
    for op in program.global_block().ops:
        for in_var_name in op.input_arg_names:
            if in_var_name not in visited_vars:
                input_vars.append(in_var_name)
                visited_vars.add(in_var_name)

        for out_var_name in op.output_arg_names:
            visited_vars.add(out_var_name)
    return input_vars


def get_outputs_of_program(program):
    output_vars = OrderedDict()
    for op in program.global_block().ops:
        list_to_ordered_dict(op.output_arg_names, output_vars)
    return list(output_vars.keys())


def prune_program(program, start_op_idx, end_op_idx):
    op_num = len(program.global_block().ops)
    if start_op_idx < 0:
        start_op_idx += op_num
    assert start_op_idx >= 0 and start_op_idx < op_num
    if end_op_idx < 0:
        end_op_idx += op_num
    assert end_op_idx >= 0 and end_op_idx <= op_num, end_op_idx
    assert start_op_idx < end_op_idx

    program = program.clone()
    for idx in range(op_num - 1, end_op_idx - 1, -1):
        program.global_block()._remove_op(idx, sync=False)
    for idx in range(start_op_idx - 1, -1, -1):
        program.global_block()._remove_op(idx, sync=False)
    program._sync_with_cpp()

    valid_vars = set()
    for op in program.global_block().ops:
        for in_var_name in op.input_arg_names:
            valid_vars.add(in_var_name)
        for out_var_name in op.output_arg_names:
            valid_vars.add(out_var_name)

    vars_to_remove = []
    for var in program.global_block().vars:
        if var not in valid_vars:
            vars_to_remove.append(var)

    for var in vars_to_remove:
        program.global_block()._remove_var(var, sync=False)
    program._sync_with_cpp()
    return program


def split_program(program, op_indices):
    """
    Split the program by op_indices.

    For examples, a program has 100 ops, and op_indices = [25, 60].
    Then the program is splitted into 3 parts, containing 25, 35 and 40
    ops respectively.

    The return values are a tuple with 3 elements: the splitted program
    list, the input var names of each splitted program, and the output
    var names of each splitted program.
    """
    assert op_indices, "op_indices cannot be empty"
    op_num = len(program.global_block().ops)
    assert op_num > 0, "program cannot be empty"

    op_indices = [idx if idx >= 0 else idx + op_num for idx in op_indices]

    if op_indices[0] != 0:
        op_indices = [0] + op_indices
    if op_indices[-1] != op_num:
        op_indices.append(op_num)

    for idx in range(len(op_indices) - 1):
        assert (
            op_indices[idx] < op_indices[idx + 1]
        ), "op_indices must be strictly sorted"

    splitted_programs = []
    for idx in range(len(op_indices) - 1):
        new_split = prune_program(program, op_indices[idx], op_indices[idx + 1])
        splitted_programs.append(new_split)

    num_split = len(splitted_programs)
    input_vars = [get_inputs_of_program(p) for p in splitted_programs]
    output_vars = [
        list_to_ordered_dict(get_outputs_of_program(p))
        for p in splitted_programs
    ]
    valid_output_vars = [OrderedDict() for _ in range(num_split)]
    valid_output_vars[-1] = output_vars[-1]
    for i in range(1, num_split):
        for in_var_name in input_vars[i]:
            for j in reversed(range(i)):
                if in_var_name in output_vars[j]:
                    valid_output_vars[j][in_var_name] = True
                    break
    valid_output_vars = [list(item.keys()) for item in valid_output_vars]
    return splitted_programs, input_vars, valid_output_vars


class OpInfo:
    def __init__(self, op):
        self.op = op
        self.no_need_buffer_slots = set()
        self.other_arg_names_set = set()

    def get_op_attrs(self):
        inputs = {}
        for input_name in self.op.input_names:
            inputs[input_name] = self.op.input(input_name)
        outputs = {}
        for output_name in self.op.output_names:
            outputs[output_name] = self.op.output(output_name)
        attrs = {}
        for attr_name in self.op.attr_names:
            attrs[attr_name] = self.op.attr(attr_name)

        return inputs, outputs, attrs

    def build_op_info(self):
        inputs, outputs, attrs = self.get_op_attrs()
        self.no_need_buffer_slots = core.infer_no_need_buffer_slots(
            self.op.type, inputs, outputs, attrs
        )
        if len(self.no_need_buffer_slots) == 0:
            return

        for slot_name in self.op.input_names:
            if slot_name in self.no_need_buffer_slots:
                continue

            for in_name in self.op.input(slot_name):
                self.other_arg_names_set.add(in_name)

        for slot_name in self.op.output_names:
            for out_name in self.op.output(slot_name):
                self.other_arg_names_set.add(out_name)

    def is_needed(self, arg_name):
        return (
            len(self.no_need_buffer_slots) == 0
            or arg_name in self.other_arg_names_set
        )


def get_skip_gc_vars(program_list: List[Program]):
    """
    Get `skip_gc_vars` for every sub_program of program_list.

    A whole_program is split up into sub_programs according to the schedule mode,
    thus a sub_program's vars might be used as the op's input of the later sub_program,
    and these vars cannot be gc after executing current sub_program.
    """

    # step1: get all non-persistable vars of every sub_program of program_list
    vars_list = [
        {
            var.name
            for var in filter(
                lambda var: not var.persistable, program.list_vars()
            )
        }
        for program in program_list
    ]

    # step2: get the `intersection_vars_list` that vars of current sub_program might be used in the later sub_program
    union_set = set()
    intersection_vars_list = [set()] * len(program_list)
    for idx, vars_set in reversed(list(enumerate(vars_list))):
        if idx < len(vars_list) - 1:
            union_set = union_set.union(vars_list[idx + 1])
        intersection_vars_list[idx] = vars_set & union_set

    # step3: Filter the vars that might be in no_need_buffer of op.
    # Reversely traversing all ops in the program_list. If op's input_args in the former vars_list,
    # and the input_args is used in op's computation, the input_args should be put in skip_gc_vars.
    skip_gc_vars = [set() for _ in range(len(program_list))]
    for ip, program in reversed(list(enumerate(program_list))):
        for op in program.global_block().ops:
            op_info = OpInfo(op)
            op_info.build_op_info()

            for in_name in op.input_arg_names:
                for i in range(0, ip):
                    if in_name not in intersection_vars_list[i]:
                        continue
                    if op_info.is_needed(in_name):
                        skip_gc_vars[i].add(in_name)

    return skip_gc_vars
