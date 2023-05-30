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

import dis

# TODO: Refactor this file


HASLOCAL_OPCODES = set(dis.haslocal)
HASFREE_OPCODES = set(dis.hasfree)
COMPARE_OPCODES = set(dis.cmp_op)
HASJREL_OPCODES = set(dis.hasjrel)
HASJABS_OPCODES = set(dis.hasjabs)
JUMP_OPCODES = HASJREL_OPCODES | HASJABS_OPCODES


def calc_offset_from_bytecode_offset(bytecode_offset: int) -> int:
    # Calculate the index from bytecode offset, because it have 2 bytes per instruction
    # TODO: Change this for Python 3.11+.
    return bytecode_offset // 2


def calc_jump_target(
    instructions: list[dis.Instruction], current_instr_idx: int
) -> int:
    """
    Handle the case where the jump target is in the middle of an extended arg.
    """
    num_instr = len(instructions)
    # For each opcode, at most three prefixal EXTENDED_ARG are allowed, so we
    # need to check at most 4 instructions.
    # See more details in https://docs.python.org/3.10/library/dis.html#opcode-EXTENDED_ARG
    for i in range(current_instr_idx, min(current_instr_idx + 4, num_instr)):
        if instructions[i].opcode != dis.EXTENDED_ARG:
            return i
    else:
        raise ValueError("Could not find jump target")


def read_write_analysis(
    instructions: list[dis.Instruction],
    current_instr_idx: int,
    stop_instr_idx: int = None,
):
    writes = set()
    reads = set()
    visited = set()

    def walk(start):
        end = len(instructions) if stop_instr_idx is None else stop_instr_idx
        for i in range(start, end):
            if i in visited:
                continue
            visited.add(i)

            instr = instructions[i]
            if instr.opcode in HASLOCAL_OPCODES | HASFREE_OPCODES:
                if (
                    instr.opname.startswith("LOAD")
                    and instr.argval not in writes
                ):
                    reads.add(instr.argval)
                elif instr.opname.startswith("STORE"):
                    writes.add(instr.argval)
            elif instr.opcode in JUMP_OPCODES:
                target_idx = calc_offset_from_bytecode_offset(instr.argval)
                target_idx = calc_jump_target(instructions, target_idx)
                # Fork to two branches, jump or not
                walk(target_idx)

    walk(current_instr_idx)
    return reads
