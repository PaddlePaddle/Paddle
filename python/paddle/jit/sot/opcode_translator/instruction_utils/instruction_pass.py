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

    def name_not_loaded_in(name, instrs):
        for instr in instrs:
            if instr.opname == "LOAD_FAST" and instr.argval == name:
                return False
        return True

    def stored_from(load_instr, instrs):
        idx = instrs.index(load_instr) - 1
        while idx >= 0:
            instr = instrs[idx]
            if (
                instr.opname == "STORE_FAST"
                and instr.argval == load_instr.argval
            ):
                return instr
            idx -= 1
        return None

    def no_more_load(load_instr, instrs):
        idx = instrs.index(load_instr) + 1
        while idx < len(instrs):
            instr = instrs[idx]
            if (
                instr.opname == "LOAD_FAST"
                and instr.argval == load_instr.argval
            ):
                return False
        return True

    # remove rename and load store
    jump_target = {
        instr.jump_to for instr in instrs if instr.jump_to is not None
    }

    modified = True
    while modified:
        modified = False
        stored_once = find_stored_once_local_vars(instrs, code_options)

        # find out all LOAD_FAST -> STORE_FAST pair
        opcode_pairs = find_related_local_opcodes(instrs, code_options)

        for load, store in opcode_pairs:
            if load in jump_target or store in jump_target:
                continue
            load_name = load.argval
            store_name = store.argval

            # if these two names are only stored once
            # it means these two name only have one value all the time
            # so we can just rename them, to delete some codes
            if load_name in stored_once and store_name in stored_once:
                instrs.remove(load)
                instrs.remove(store)
                if load_name != store_name:
                    for instr in instrs:
                        if (
                            instr.opname in ("LOAD_FAST", "STORE_FAST")
                            and instr.argval == store_name
                        ):
                            instr.argval = load_name
                modified = True

            # if
            #       LOAD A
            #       STORE B
            # A or B is not stored only once (maybe it is input)
            # we give a more general way to simplify the codes:
            #
            # if A will not be loaded again after STORE B, it means we can move STORE B ahead, like:
            #       STORE A             ->          STORE B
            #       ...                             ...
            #       LOAD  A             ->          LOAD B
            #       ...                             ...
            #       LOAD A              ->          ---- (rm)
            #       STORE B                         ---- (rm)
            # so we can rename the rest LOAD A below as LOAD B
            # Notice, to do this transform, we must make sure that there is not LOAD B exist
            # between STORE A and STORE B, for B should be another value in this range

            elif no_more_load(load, instrs):
                last_store = stored_from(load, instrs)
                if last_store is not None:
                    code_range = instrs[
                        instrs.index(last_store) : instrs.index(store)
                    ]
                else:
                    code_range = instrs[: instrs.index(store)]
                if name_not_loaded_in(store_name, code_range):
                    last_store.argval = store_name
                    instrs.remove(load)
                    instrs.remove(store)
                    for instr in instrs[instrs.index(last_store) :]:
                        if (
                            instr.opname in ("LOAD_FAST", "STORE_FAST")
                            and instr.argval == load_name
                        ):
                            instr.argval = store_name

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
