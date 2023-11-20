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


def apply_instr_pass(instrs, code_options):
    supported_passes = (remove_temporary_local_var_pass,)

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
                stored_vars[name] += 1
            else:
                stored_vars[name] = 1

    stored_once = set([name for name, count in stored_vars if count == 1])
    return stored_once


def find_related_local_opcodes(instrs, code_options):
    """
    find out the opcode pairs consist with LOAD_FAST and STORE_FAST
    """
    pass


def remove_temporary_local_var_pass(instrs, code_options):
    """
    This question is extremely complex, so we just simplify it as
    'remove renames which is between var names who only stored once'
    and we only consider the local vars.
    """
    stored_once = find_stored_once_local_vars(instrs, code_options)
    opcode_pairs = find_related_local_opcodes(instrs, code_options)

    for opcode1, opcode2 in opcode_pairs:
        if opcode1.argval == opcode2.argval:
            instrs.remove(opcode1)
            instrs.remove(opcode2)
        elif opcode1.argval in stored_once and opcode2.argval in stored_once:
            instrs.remove(opcode1)
            instrs.remove(opcode2)
            for instr in instrs:
                if (
                    instr.opname in ("LOAD_FAST", "STORE_FAST")
                    and instr.argval == opcode1.argval
                ):
                    instr.argval = opcode2.argval
