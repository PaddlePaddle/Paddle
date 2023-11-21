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

from .stack_analyse import StackAnalyser


def apply_instr_pass(instrs, code_options):
    supported_passes = (remove_load_store_pass,)

    for instr_pass in supported_passes:
        instr_pass(instrs, code_options)


def find_stored_once_local_vars(instrs, code_options):
    """
    find out the local var names which is only stored once
    """
    stored_vars = {}

    # The input vars are considered as stored at the beginning
    input_names = code_options['co_varnames'][: code_options['co_argcount']]

    for name in input_names:
        stored_vars[name] = 1

    for instr in instrs:
        if instr.opname == "STORE_FAST":
            if instr.argval in stored_vars:
                stored_vars[instr.argval] += 1
            else:
                stored_vars[instr.argval] = 1

    stored_once = {name for name, count in stored_vars.items() if count == 1}
    return stored_once


def find_loaded_once_local_vars(instrs, code_options):
    """
    find out the local var names which is only stored once
    """
    loaded_vars = {}
    for instr in instrs:
        if instr.opname == "STORE_FAST":
            if instr.argval in loaded_vars:
                loaded_vars[instr.argval] += 1
            else:
                loaded_vars[instr.argval] = 1

    stored_once = {name for name, count in loaded_vars.items() if count == 1}
    return stored_once


def find_related_local_opcodes(instrs, code_options):
    """
    find out the opcode pairs consist with LOAD_FAST and STORE_FAST
    """
    stack = []
    opcode_pairs = []
    for instr in instrs:
        if instr.opname == "LOAD_FAST":
            stack.append(instr)
        elif instr.opname == "STORE_FAST":
            if len(stack) > 0 and stack[-1] is not None:
                opcode_pairs.append((stack[-1], instr))
            stack.pop()
        else:
            try:
                pop_n, push_n = StackAnalyser().stack_effect(instr)
                if pop_n == 0:
                    stack.extend([None] * push_n)
                else:
                    stack = stack[:-pop_n] + [None] * push_n
            except AttributeError:
                break

    return opcode_pairs


def remove_load_store_pass(instrs, code_options):
    """
    This question is extremely complex, so we just simplify it as
    'remove renames which is between var names who only stored once'
    and we only consider the local vars.
    """

    # remove rename and load store
    stored_once = find_stored_once_local_vars(instrs, code_options)
    jump_target = {
        instr.jump_to for instr in instrs if instr.jump_to is not None
    }

    modified = True
    while modified:
        modified = False
        opcode_pairs = find_related_local_opcodes(instrs, code_options)

        for opcode1, opcode2 in opcode_pairs:
            if opcode1 in jump_target or opcode2 in jump_target:
                continue
            if opcode1.argval in stored_once and opcode2.argval in stored_once:
                instrs.remove(opcode1)
                instrs.remove(opcode2)
                if opcode1.argval != opcode2.argval:
                    for instr in instrs:
                        if (
                            instr.opname in ("LOAD_FAST", "STORE_FAST")
                            and instr.argval == opcode2.argval
                        ):
                            instr.argval = opcode1.argval
                modified = True

    # remove store load
    loaded_once = find_loaded_once_local_vars(instrs, code_options)

    modified = True
    while modified:
        modified = False

        idx = 0
        while idx + 1 < len(instrs):
            opcode1 = instrs[idx]
            opcode2 = instrs[idx + 1]

            if (
                opcode1 not in jump_target
                and opcode2 not in jump_target
                and opcode1.opname == "STORE_FAST"
                and opcode2.opname == "LOAD_FAST"
                and opcode1.argval == opcode2.argval
                and opcode1.argval in loaded_once
            ):
                instrs.remove(opcode1)
                instrs.remove(opcode2)
                modified = True
            else:
                idx += 1
