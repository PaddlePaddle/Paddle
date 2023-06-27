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


class OpInOutInfo:
    """
    Record unused buffer input_vars of op and other var_names except unused buffer input_vars
    """

    def __init__(self):
        self._is_build = False
        self._no_need_buffer_slots = set()
        self._other_arg_names_set = set()

    @property
    def is_build(self):
        return self._is_build

    def _get_op_attrs(self, op):
        inputs = {}
        for input_name in op.input_names:
            inputs[input_name] = op.input(input_name)
        outputs = {}
        for output_name in op.output_names:
            outputs[output_name] = op.output(output_name)
        attrs = {}
        for attr_name in op.attr_names:
            attrs[attr_name] = op.attr(attr_name)

        return inputs, outputs, attrs

    def build_info(self, op):
        inputs, outputs, attrs = self._get_op_attrs(op)
        self._no_need_buffer_slots = core.infer_no_need_buffer_slots(
            op.type, inputs, outputs, attrs
        )
        if len(self._no_need_buffer_slots) == 0:
            return

        for slot_name in op.input_names:
            if slot_name in self._no_need_buffer_slots:
                continue

            for in_name in op.input(slot_name):
                self._other_arg_names_set.add(in_name)

        for slot_name in op.output_names:
            for out_name in op.output(slot_name):
                self._other_arg_names_set.add(out_name)

        self._is_build = True

    def is_needed(self, arg_name):
        return (
            len(self._no_need_buffer_slots) == 0
            or arg_name in self._other_arg_names_set
        )


def var_can_be_deleted(var_name, program):
    var = program.global_block()._find_var_recursive(var_name)
    if var is None or var.persistable:
        return False

    return var.type in [
        core.VarDesc.VarType.LOD_TENSOR,
        core.VarDesc.VarType.SELECTED_ROWS,
        core.VarDesc.VarType.LOD_TENSOR_ARRAY,
    ]


def get_skip_gc_vars(program_list: List[Program]):
    """
    Get `skip_gc_vars` for every sub_program of program_list.

    A whole_program is split up into sub_programs according to the schedule mode,
    thus a sub_program's vars might be used as the op's input of the later sub_program,
    and these vars cannot be gc after executing current sub_program.
    """

    # step1: Get all vars of every sub_program of program_list that are non-persistable and not in op's no_need_buffer.
    vars_list = [set() for _ in range(len(program_list))]
    for ip, program in enumerate(program_list):
        for op in program.global_block().ops:
            op_info = OpInOutInfo()
            for in_name in op.input_arg_names:
                if not var_can_be_deleted(in_name, program):
                    continue

                if not op_info.is_build:
                    op_info.build_info(op)

                if op_info.is_needed(in_name):
                    vars_list[ip].add(in_name)

            for out_name in op.output_arg_names:
                if var_can_be_deleted(out_name, program):
                    vars_list[ip].add(out_name)

    # step2: get the `skip_gc_vars` that vars of current sub_program might be used in the later sub_program
    union_set = set()
    skip_gc_vars = [set()] * len(program_list)
    for idx, vars_set in reversed(list(enumerate(vars_list))):
        if idx < len(vars_list) - 1:
            union_set = union_set.union(vars_list[idx + 1])
        skip_gc_vars[idx] = vars_set & union_set

    return skip_gc_vars
