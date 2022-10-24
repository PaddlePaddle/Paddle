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
