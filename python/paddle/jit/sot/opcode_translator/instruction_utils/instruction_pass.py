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

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from paddle.jit.sot.utils import log, log_do

from ...utils import InnerError
from .instruction_utils import instrs_info
from .stack_analyse import StackAnalyser

if TYPE_CHECKING:
    from .instruction_utils import Instruction


def apply_instr_pass(instrs: list[Instruction], code_options):
    log(4, f"[Opcode Pass]: Original New Code {code_options['co_name']}:\n")
    log_do(4, lambda: print(instrs_info(instrs)))
    supported_passes = [
        remove_load_store_pass,
        remove_duplicate_resume,
        check_precall_followed_by_call,
    ]

    if sys.version_info >= (3, 12):
        supported_passes.append(check_for_iter_jump_to)

    for instr_pass in supported_passes:
        instr_pass(instrs, code_options)

    log(
        4,
        f"[Opcode Pass]: New Code After Opcode Pass {code_options['co_name']}:\n",
    )
    log_do(4, lambda: print(instrs_info(instrs)))


def find_stored_once_local_vars(instrs: list[Instruction], code_options):
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


def find_loaded_once_local_vars(instrs: list[Instruction], code_options):
    """
    find out the local var names which is only stored once
    """
    loaded_vars = {}
    for instr in instrs:
        if instr.opname in ["LOAD_FAST", "LOAD_FAST_CHECK"]:
            if instr.argval in loaded_vars:
                loaded_vars[instr.argval] += 1
            else:
                loaded_vars[instr.argval] = 1

    loaded_once = {name for name, count in loaded_vars.items() if count == 1}
    return loaded_once


def find_related_local_opcodes(instrs: list[Instruction], code_options):
    """
    find out the opcode pairs consist with LOAD_FAST and STORE_FAST and LOAD_FAST_CHECK
    """
    stack = []
    opcode_pairs = []
    for instr in instrs:
        if instr.opname in ["LOAD_FAST", "LOAD_FAST_CHECK"]:
            stack.append(instr)
        elif instr.opname == "STORE_FAST":
            if len(stack) > 0 and stack[-1] is not None:
                opcode_pairs.append((stack[-1], instr))
            stack.pop()
        elif "ROT" in instr.opname or "DUP" in instr.opname:
            return []
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


def remove_load_store_pass(instrs: list[Instruction], code_options):
    """
    This question is extremely complex, so we just simplify it as
    'remove renames which is between var names who only stored once'
    and we only consider the local vars.
    """

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

    def code_exist(opname, argval, instrs):
        for instr in instrs:
            if instr.opname == opname and instr.argval == argval:
                return True
        return False

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

        for load_a, store_b in opcode_pairs:
            if load_a in jump_target or store_b in jump_target:
                continue
            a_name = load_a.argval
            b_name = store_b.argval

            # if these two names are only stored once
            # it means these two name only have one value all the time
            # so we can just rename them, to delete some codes
            if a_name in stored_once and b_name in stored_once:
                instrs.remove(load_a)
                instrs.remove(store_b)
                if a_name != b_name:
                    for instr in instrs:
                        if (
                            instr.opname
                            in ("LOAD_FAST_CHECK", "LOAD_FAST", "STORE_FAST")
                            and instr.argval == b_name
                        ):
                            instr.argval = a_name
                            instr.arg = load_a.arg
                modified = True

            # if
            #       LOAD A
            #       STORE B
            # A or B is not stored only once (maybe it is input)
            # we give a more general way to simplify the codes
            #
            # if A will not be loaded again after (6)STORE B, it means we can move (6)STORE B ahead to (1)STORE A
            # TIP: there is no more STORE A between (1) and (5)
            #  (1)      STORE A             ->          STORE B
            #           ...                             ...
            #  (2)      LOAD A              ->          LOAD B
            #           ...
            #  (3)      LOAD B              ->          not support
            #           ...
            #  (4)      STORE B             ->          not support
            #           ...                             ...
            #  (5)      LOAD A              ->          ---- (rm)
            #  (6)      STORE B                         ---- (rm)
            #           ...
            #  (7)      STORE B
            #  (8)      LOAD A
            # so we can rename the rest LOAD A below as LOAD B
            #
            # What changed:
            #   1. if (4) exist, B changed:
            #       (1) ~ (4), (6) ~
            #   2. if (4) not exist, B changed:
            #       (1), (6)
            #   3. A changed:
            #       (1) ~
            #
            # To do this transform, we should make sure
            #   1. (4) is not exist in (1) ~ (5): it is too complex
            #   2. (3) is not exist in (1) ~ (5): load B in the range that B value is changed
            #   3. (7) (8) is not exist in (6)~: load A in range that A value is changed, if we load B instead, but B also changed
            #       we can simplify this as "no more LOAD A after (6)"
            else:
                last_store_a = stored_from(load_a, instrs)
                if last_store_a is None:
                    # if last store a just not exist, we can not do this transform
                    continue

                last_store_idx = instrs.index(last_store_a)
                code_range = instrs[last_store_idx : instrs.index(store_b)]
                if (
                    not code_exist("STORE_FAST", b_name, code_range)
                    and not code_exist("LOAD_FAST_CHECK", b_name, code_range)
                    and not code_exist("LOAD_FAST", b_name, code_range)
                    and not code_exist(
                        "LOAD_FAST_CHECK",
                        a_name,
                        instrs[instrs.index(store_b) :],
                    )
                    and not code_exist(
                        "LOAD_FAST", a_name, instrs[instrs.index(store_b) :]
                    )
                ):
                    last_store_a.argval = b_name
                    last_store_a.arg = store_b.arg
                    instrs.remove(load_a)
                    instrs.remove(store_b)
                    for instr in instrs[last_store_idx:]:
                        if (
                            instr.opname
                            in ("LOAD_FAST_CHECK", "LOAD_FAST", "STORE_FAST")
                            and instr.argval == a_name
                        ):
                            instr.argval = b_name
                            instr.arg = store_b.arg

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
                and opcode2.opname == "LOAD_FAST_CHECK"
                and opcode1.argval == opcode2.argval
                and opcode1.argval in loaded_once
            ):
                instrs.remove(opcode1)
                instrs.remove(opcode2)
                modified = True
            else:
                idx += 1


def remove_duplicate_resume(instrs: list[Instruction], code_options):
    resumes = list(filter(lambda instr: instr.opname == "RESUME", instrs))
    if not resumes:
        return
    for resume in resumes[1:]:
        instrs.remove(resume)


def check_precall_followed_by_call(instrs: list[Instruction], code_options):
    """
    PRECALL should be followed by CALL, otherwise it will cause a segmentation fault
    """
    for instr, next_instr in zip(instrs[:-1], instrs[1:]):
        if instr.opname == "PRECALL" and next_instr.opname != "CALL":
            raise InnerError(
                f"PRECALL is not followed by CALL in {code_options['co_name']}"
            )


def check_for_iter_jump_to(instrs: list[Instruction], code_options):
    """
    Check if the `jump_to` of FOR_ITER is END_FOR, in Python3.12+
    """
    for instr in instrs:
        if instr.opname == "FOR_ITER":
            assert instr.jump_to is not None
            if instr.jump_to.opname != "END_FOR":
                raise InnerError("FOR_ITER jump_to is not END_FOR")
