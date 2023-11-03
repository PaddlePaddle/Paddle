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
from enum import Enum

from ...utils import InnerError, OrderedSet
from .instruction_utils import Instruction
from .opcode_info import ALL_JUMP, HAS_FREE, HAS_LOCAL, UNCONDITIONAL_JUMP


@dataclasses.dataclass
class State:
    reads: OrderedSet[str]
    writes: OrderedSet[str]
    visited: OrderedSet[int]


def is_read_opcode(opname):
    if opname in [
        "LOAD_FAST",
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


def analysis_inputs(
    instructions: list[Instruction],
    current_instr_idx: int,
    stop_instr_idx: int | None = None,
) -> OrderedSet[str]:
    """
    Analyze the inputs of the instructions from current_instr_idx to stop_instr_idx.

    Args:
        instructions (list[Instruction]): The instructions to analyze.
        current_instr_idx (int): The index of the current instruction.
        stop_instr_idx (int | None, optional): The index of the instruction to stop. Defaults to None.
            If None, the analysis will stop at the end of the instructions.

    Returns:
        set[str]: The analysis result.
    """
    root_state = State(OrderedSet(), OrderedSet(), OrderedSet())

    def fork(
        state: State, start: int, jump: bool, jump_target: int
    ) -> OrderedSet[str]:
        new_start = start + 1 if not jump else jump_target
        new_state = State(
            OrderedSet(state.reads),
            OrderedSet(state.writes),
            OrderedSet(state.visited),
        )
        return walk(new_state, new_start)

    def walk(state: State, start: int) -> OrderedSet[str]:
        end = len(instructions) if stop_instr_idx is None else stop_instr_idx
        for i in range(start, end):
            if i in state.visited:
                return state.reads
            state.visited.add(i)

            instr = instructions[i]
            if instr.opname in HAS_LOCAL | HAS_FREE:
                if is_read_opcode(instr.opname) and instr.argval not in (
                    state.writes
                ):
                    state.reads.add(instr.argval)
                elif is_write_opcode(instr.opname):
                    state.writes.add(instr.argval)
            elif instr.opname in ALL_JUMP:
                assert instr.jump_to is not None
                target_idx = instructions.index(instr.jump_to)
                # Fork to two branches, jump or not
                jump_branch = fork(state, i, True, target_idx)
                not_jump_branch = (
                    fork(state, i, False, target_idx)
                    if instr.opname not in UNCONDITIONAL_JUMP
                    else OrderedSet()
                )
                return jump_branch | not_jump_branch
            elif instr.opname == "RETURN_VALUE":
                return state.reads
        return state.reads

    return walk(root_state, current_instr_idx)


@dataclasses.dataclass
class SpaceState:
    reads: dict[str, Space]
    writes: dict[str, Space]
    visited: OrderedSet[int]

    def __or__(self, other):
        reads = {}
        reads.update(other.reads)
        reads.update(self.reads)
        writes = {}
        writes.update(other.writes)
        writes.update(self.writes)
        return SpaceState(reads, writes, OrderedSet())


class Space(Enum):
    locals = 1
    globals = 2
    cells = 3
    all = 4


def get_space(opname: str):
    if "FAST" in opname:
        return Space.locals
    elif "GLOBAL" in opname:
        return Space.globals
    elif "DEREF" in opname or "CLOSURE" in opname:
        return Space.cells
    elif "NAME" in opname:
        return Space.all
    else:
        raise InnerError(f"Unknown space for {opname}")


def analysis_used_names_with_space(
    instructions: list[Instruction],
    start_instr_idx: int,
    stop_instr_idx: int | None = None,
):
    root_state = SpaceState({}, {}, OrderedSet())

    def fork(
        state: SpaceState, start: int, jump: bool, jump_target: int
    ) -> SpaceState:
        new_start = start + 1 if not jump else jump_target
        new_state = SpaceState(
            dict(state.reads),
            dict(state.writes),
            OrderedSet(state.visited),
        )
        return walk(new_state, new_start)

    def walk(state: SpaceState, start: int) -> SpaceState:
        end = len(instructions) if stop_instr_idx is None else stop_instr_idx
        for i in range(start, end):
            if i in state.visited:
                return state
            state.visited.add(i)

            instr = instructions[i]
            if instr.opname in HAS_LOCAL | HAS_FREE:
                if is_read_opcode(instr.opname) and instr.argval not in (
                    state.writes
                ):
                    space = get_space(instr.opname)
                    state.reads[instr.argval] = space
                elif is_write_opcode(instr.opname):
                    space = get_space(instr.opname)
                    state.writes[instr.argval] = space
            elif instr.opname in ALL_JUMP:
                assert instr.jump_to is not None
                target_idx = instructions.index(instr.jump_to)
                # Fork to two branches, jump or not
                jump_branch = fork(state, i, True, target_idx)
                not_jump_branch = (
                    fork(state, i, False, target_idx)
                    if instr.opname not in UNCONDITIONAL_JUMP
                    else SpaceState({}, {}, OrderedSet())
                )
                return jump_branch | not_jump_branch
            elif instr.opname == "RETURN_VALUE":
                return state
        return state

    state = walk(root_state, start_instr_idx)
    all_used_vars = {}
    all_used_vars.update(state.writes)
    all_used_vars.update(state.reads)
    return all_used_vars
