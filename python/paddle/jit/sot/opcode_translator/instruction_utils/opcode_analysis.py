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

import dataclasses
from typing import TYPE_CHECKING

from paddle.jit.utils import OrderedSet

from .opcode_info import (
    ALL_JUMP,
    HAS_FREE,
    HAS_LOCAL,
    UNCONDITIONAL_JUMP,
)

if TYPE_CHECKING:
    from .instruction_utils import Instruction


@dataclasses.dataclass
class NameRecorder:
    reads: OrderedSet[str]
    writes: OrderedSet[str]

    def __or__(self, other):
        reads = self.reads | other.reads
        writes = self.writes | other.writes
        return NameRecorder(reads, writes)


def is_read_opcode(opname):
    if opname in [
        "LOAD_FAST",
        "LOAD_FAST_CHECK",
        "LOAD_DEREF",
        "LOAD_NAME",
        "LOAD_GLOBAL",
        "LOAD_CLOSURE",
    ]:
        return True
    if opname in (
        "DELETE_FAST",
        "DELETE_DEREF",
        "DELETE_NAME",
        "DELETE_GLOBAL",
    ):
        return True
    return False


def is_write_opcode(opname):
    if opname in ["STORE_FAST", "STORE_NAME", "STORE_DEREF", "STORE_GLOBAL"]:
        return True
    if opname in (
        "DELETE_FAST",
        "DELETE_DEREF",
        "DELETE_NAME",
        "DELETE_GLOBAL",
    ):
        return True
    return False


def analysis_used_names(
    instructions: list[Instruction],
    current_instr_idx: int,
    stop_instr_idx: int | None = None,
) -> tuple[OrderedSet[str], OrderedSet[str]]:
    """
    Analyze the inputs of the instructions from current_instr_idx to stop_instr_idx.

    Args:
        instructions (list[Instruction]): The instructions to analyze.
        current_instr_idx (int): The index of the current instruction.
        stop_instr_idx (int | None, optional): The index of the instruction to stop. Defaults to None.
            If None, the analysis will stop at the end of the instructions.

    Returns:
        State: The analysis result.
    """
    name_recorder = NameRecorder(OrderedSet(), OrderedSet())

    # start idx and writes names can decide the analysis result below
    # so, just check the pair of (idx, writes), to skip repeat simulation
    # (writes can decide if a name should be add to reads)
    # one idx can has multi writes for whom is not subset with each other
    # if A is subset of B, we just record A, simulate A might add more reads
    visited_states = {}

    def check_and_update_visited_states(idx, writes):
        writes = set(writes)

        if idx in visited_states:
            history = visited_states[idx]
            for record in history:
                if record.issubset(writes):
                    return True
                elif writes.issubset(record):
                    history.remove(record)
                    history.append(writes)
                    return False
        else:
            visited_states[idx] = [writes]

        return False

    def fork(
        name_recorder: NameRecorder, start: int, jump: bool, jump_target: int
    ) -> NameRecorder:
        new_start = start + 1 if not jump else jump_target
        new_state = NameRecorder(
            OrderedSet(name_recorder.reads),
            OrderedSet(name_recorder.writes),
        )
        return walk(new_state, new_start)

    def walk(name_recorder: NameRecorder, start: int) -> NameRecorder:
        end = len(instructions) if stop_instr_idx is None else stop_instr_idx
        for i in range(start, end):
            if check_and_update_visited_states(i, name_recorder.writes):
                return name_recorder

            instr = instructions[i]
            if instr.opname in HAS_LOCAL | HAS_FREE:
                if is_read_opcode(instr.opname) and instr.argval not in (
                    name_recorder.writes
                ):
                    name_recorder.reads.add(instr.argval)
                elif is_write_opcode(instr.opname):
                    name_recorder.writes.add(instr.argval)
            elif instr.opname in ALL_JUMP:
                assert instr.jump_to is not None
                target_idx = instructions.index(instr.jump_to)
                # Fork to two branches, jump or not
                jump_branch = fork(name_recorder, i, True, target_idx)
                not_jump_branch = (
                    fork(name_recorder, i, False, target_idx)
                    if instr.opname not in UNCONDITIONAL_JUMP
                    else NameRecorder(OrderedSet(), OrderedSet())
                )
                return jump_branch | not_jump_branch
            elif instr.opname == "RETURN_VALUE":
                return name_recorder
        return name_recorder

    name_recorder = walk(name_recorder, current_instr_idx)
    return name_recorder.reads, name_recorder.writes
