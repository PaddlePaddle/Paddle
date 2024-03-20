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
import dis
import sys
from enum import Enum
from typing import TYPE_CHECKING, Any

from ...utils import InnerError
from .opcode_info import (
    ABS_JUMP,
    ALL_JUMP,
    PYOPCODE_CACHE_SIZE,
    REL_BWD_JUMP,
    REL_JUMP,
)

if TYPE_CHECKING:
    import types


@dataclasses.dataclass
class Instruction:
    opcode: int
    opname: str
    arg: int | None
    argval: Any
    offset: int | None = None
    starts_line: int | None = None
    is_jump_target: bool = False
    jump_to: Instruction | None = None
    is_generated: bool = True

    # for analysis EXTENDED_ARG
    first_ex_arg: Instruction | None = None
    ex_arg_for: Instruction | None = None

    # used in modify_extended_args
    def __hash__(self):
        return id(self)

    def __eq__(self, instr):
        return id(self) == id(instr)


def gen_instr(name, arg=None, argval=None, gened=True, jump_to=None):
    return Instruction(
        opcode=dis.opmap[name],
        opname=name,
        arg=arg,
        argval=argval,
        is_generated=gened,
        jump_to=jump_to,
    )


def convert_instruction(instr: dis.Instruction) -> Instruction:
    """
    Converts a disassembled instruction to a customized Instruction object.

    Args:
        instr (dis.Instruction): The disassembled instruction.

    Returns:
        Instruction: A customized Instruction object.
    """
    return Instruction(
        instr.opcode,
        instr.opname,
        instr.arg,
        instr.argval,
        instr.offset,
        instr.starts_line,
        instr.is_jump_target,
        jump_to=None,
        is_generated=False,
    )


def get_instructions(code: types.CodeType) -> list[Instruction]:
    """
    Returns parsed instructions from the given code object and exclude
    any opcodes that contain `EXTENDED_ARG`.

    Args:
        code (types.CodeType): The code object to extract instructions from.

    Returns:
        list[Instruction]: A list of Instruction objects representing the
            bytecode instructions in the code object.
    """
    # instrs do not contain EXTENDED_ARG
    instrs = list(map(convert_instruction, dis.get_instructions(code)))
    for instr in instrs:
        if instr.opname in ALL_JUMP:
            origin_jump_target = calc_offset_from_bytecode_offset(
                instr.argval, instrs
            )
            jump_offset = origin_jump_target

            while instrs[jump_offset].opname == "EXTENDED_ARG":
                jump_offset += 1

            if origin_jump_target != jump_offset:
                # copy infos from EXTENDED_ARG to other opcode

                if instrs[origin_jump_target].is_jump_target:
                    instrs[jump_offset].is_jump_target = instrs[
                        origin_jump_target
                    ].is_jump_target
                if instrs[origin_jump_target].starts_line:
                    instrs[jump_offset].starts_line = instrs[
                        origin_jump_target
                    ].starts_line

            instr.jump_to = instrs[jump_offset]

    # if the origin opcode contains EXTENDED_ARG, it should be like:
    #     >>  EXTENDED_ARG 1
    #         XX 388    <-  256 + 132
    # filter all EXTENDED_ARG here
    instrs = [x for x in instrs if x.opname != "EXTENDED_ARG"]
    return instrs


def modify_instrs(instructions: list[Instruction]) -> None:
    """
    Modifies the given list of instructions. It contains three steps:

    1. reset offset
    2. relocate jump target
    3. add EXTENDED_ARG instruction if needed

    Args:
        instructions (list): The list of Instruction objects representing bytecode instructions.

    Returns:
        None
    """
    modify_completed = False
    while not modify_completed:
        reset_offset(instructions)
        relocate_jump_target(instructions)
        modify_completed = modify_extended_args(instructions)


def reset_offset(instructions: list[Instruction]) -> None:
    """
    Resets the offset for each instruction in the list.

    Args:
        instructions (list): The list of Instruction objects representing bytecode instructions.

    Returns:
        None
    """
    from ..executor.pycode_generator import get_instruction_size

    if sys.version_info >= (3, 11):
        current_offset = 0
        for instr in instructions:
            instr.offset = current_offset
            current_offset += get_instruction_size(instr)
        return
    for idx, instr in enumerate(instructions):
        instr.offset = idx * 2


def correct_jump_direction(instr: Instruction, arg: int) -> Instruction:
    """
    Corrects the jump direction of the given instruction.
    NOTE(zrr1999): In Python 3.11, JUMP_ABSOLUTE is removed, so python generates JUMP_FORWARD or JUMP_BACKWARD instead,
    but in for loop breakgraph, we reuse JUMP_BACKWARD to jump forward, so we need to change it to JUMP_FORWARD.

    Args:
        instr (Instruction): The instruction to be corrected.
    """
    if instr.opname in ABS_JUMP:
        instr.arg = arg
        return instr
    elif instr.opname in REL_JUMP:
        if arg < 0:
            if instr.opname in REL_BWD_JUMP:
                forward_op_name = instr.opname.replace("BACKWARD", "FORWARD")
                if forward_op_name not in dis.opmap:
                    raise InnerError(f"Unknown jump type {instr.opname}")
                instr.opname = forward_op_name
                instr.opcode = dis.opmap[forward_op_name]
            else:  # instr.opname in REL_FWD_JUMP
                backward_op_name = instr.opname.replace("FORWARD", "BACKWARD")
                if backward_op_name not in dis.opmap:
                    raise InnerError(f"Unknown jump type {instr.opname}")
                instr.opname = backward_op_name
                instr.opcode = dis.opmap[backward_op_name]
            instr.arg = -arg
        else:
            instr.arg = arg
        return instr
    else:
        raise ValueError(f"unknown jump type: {instr.opname}")


def relocate_jump_target(instructions: list[Instruction]) -> None:
    """
    If a jump instruction is found, this function will adjust the jump targets based on the presence of EXTENDED_ARG instructions.
    If an EXTENDED_ARG instruction exists for the jump target, use its offset as the new target.

    Args:
        instructions (list): The list of Instruction objects representing bytecode instructions.

    Returns:
        None
    """
    extended_arg = []
    for instr in instructions:
        if instr.opname == "EXTENDED_ARG":
            extended_arg.append(instr)
            continue

        if instr.opname in ALL_JUMP:
            assert instr.jump_to is not None
            assert instr.offset is not None
            # if jump target has extended_arg, should jump to the first extended_arg opcode
            jump_target = (
                instr.jump_to.offset
                if instr.jump_to.first_ex_arg is None
                else instr.jump_to.first_ex_arg.offset
            )
            assert jump_target is not None

            if instr.opname in ABS_JUMP:
                new_arg = jump_target
            else:  # instr.opname in REL_JUMP
                cache_size = PYOPCODE_CACHE_SIZE.get(instr.opname, 0)
                new_arg = jump_target - (2 * cache_size) - instr.offset - 2
                if instr.opname in REL_BWD_JUMP:
                    new_arg = -new_arg

            if sys.version_info >= (3, 10):
                new_arg //= 2
            correct_jump_direction(instr, new_arg)
            assert instr.arg is not None
            if extended_arg:
                instr.arg &= 0xFF
                new_arg = new_arg >> 8
                for ex in reversed(extended_arg):
                    ex.arg = new_arg & 0xFF
                    new_arg = new_arg >> 8

                # need more extended_args instr
                # set arg in the first extended_arg
                if new_arg > 0:
                    extended_arg[0].arg += new_arg << 8
        extended_arg.clear()


def modify_extended_args(instructions: list[Instruction]) -> bool:
    """
    This function replaces any instruction with an argument greater than or equal to 256 with one or more EXTENDED_ARG instructions.

    Args:
        instructions (list): The list of Instruction objects representing bytecode instructions.

    Returns:
        bool: True if the modification is completed, False otherwise.
    """

    modify_completed = True
    extend_args_record = {}
    for instr in instructions:
        if instr.arg and instr.arg >= 256:  # more than one byte
            _instrs = [
                instr
            ]  # replace instr with _instrs later (it is a set of instrs), all operations will be recorded in extend_args_record
            val = instr.arg
            instr.arg = val & 0xFF
            val = val >> 8
            while val > 0:
                _instrs.append(gen_instr("EXTENDED_ARG", arg=val & 0xFF))
                val = val >> 8

            extend_args_record.update({instr: list(reversed(_instrs))})

    if extend_args_record:
        # if new EXTENDED_ARG inserted, we need update offset and jump target
        modify_completed = False

        def bind_ex_arg_with_instr(ex_arg, instr):
            # move opcode info to EXTENDED_ARG
            ex_arg.starts_line = instr.starts_line
            instr.starts_line = None
            ex_arg.is_jump_target = instr.is_jump_target
            instr.is_jump_target = False

            if instr.ex_arg_for is not None:
                # instr is also an ex_arg for another instr
                instr.ex_arg_for.first_ex_arg = ex_arg
                ex_arg.ex_arg_for = instr.ex_arg_for
                instr.ex_arg_for = None
            else:
                instr.first_ex_arg = ex_arg
                ex_arg.ex_arg_for = instr

        for key, val in extend_args_record.items():
            bind_ex_arg_with_instr(val[0], key)
            replace_instr(instructions, instr=key, new_instr=val)

    return modify_completed


def modify_vars(instructions: list[Instruction], code_options):
    co_names = code_options['co_names']
    co_varnames = code_options['co_varnames']
    co_freevars = code_options['co_freevars']
    for instrs in instructions:
        if instrs.opname in ['LOAD_FAST', 'LOAD_FAST_CHECK', 'STORE_FAST']:
            assert (
                instrs.argval in co_varnames
            ), f"`{instrs.argval}` not in {co_varnames}"
            instrs.arg = co_varnames.index(instrs.argval)
        elif instrs.opname == "LOAD_DEREF" or instrs.opname == "STORE_DEREF":
            if sys.version_info >= (3, 11):
                namemap = co_varnames + co_freevars
                assert (
                    instrs.argval in namemap
                ), f"`{instrs.argval}` not in {namemap}"
                instrs.arg = namemap.index(instrs.argval)


def calc_offset_from_bytecode_offset(
    bytecode_offset: int,
    instructions: list[dis.Instruction] | list[Instruction],
) -> int:
    """
    Calculate the index from bytecode offset, because it have 2 bytes per instruction (for Python <= 3.10).

    Args:
        bytecode_offset (int): The bytecode offset of the instruction.

    Returns:
        int: The index of the instruction in the instruction list.
    """

    if sys.version_info >= (3, 11):
        instruction_offsets = [x.offset for x in instructions]
        return instruction_offsets.index(bytecode_offset)
    return bytecode_offset // 2


def replace_instr(instructions, instr, new_instr):
    idx = instructions.index(instr)
    instructions[idx : idx + 1] = new_instr


def instrs_info(instrs, mark=None, range=None, want_str=True):
    ret = []
    start = -1
    end = 1000000
    if mark is not None and range is not None:
        start = mark - range
        end = mark + range + 1
    for idx, instr in enumerate(instrs):
        if idx < start or idx >= end:
            continue
        if instr.starts_line is not None:
            ret.append("")
        ret.append(
            "{line:<8s}{is_jump_target:>2s}{offset:>4d} {opname:<30s}{arg:<4s}{argval:<40s}{mark}".format(
                line=str(instr.starts_line) if instr.starts_line else "",
                is_jump_target=">>" if instr.is_jump_target else "  ",
                offset=instr.offset
                if instr.offset or instr.offset == 0
                else -1,
                opname=instr.opname,
                arg=str(instr.arg) if instr.arg is not None else "",
                argval=f"({instr.argval})" if instr.argval else "",
                mark="",
            )
        )
        if idx == mark:
            ret[-1] = "\033[31m" + ret[-1] + "\033[0m"
    if want_str:
        return "\n".join(ret)
    return ret


def calc_stack_effect(instr: Instruction, *, jump: bool | None = None) -> int:
    """
    Gets the stack effect of the given instruction. In Python 3.11, the stack effect of `CALL` is -1,
    refer to https://github.com/python/cpython/blob/3.11/Python/compile.c#L1123-L1124.

    Args:
        instr: The instruction.

    Returns:
        The stack effect of the instruction.

    """
    if sys.version_info[:2] == (3, 11):
        if instr.opname == "PRECALL":
            return 0
        elif instr.opname == "CALL":
            # NOTE(zrr1999): push_n = 1, pop_n = oparg + 2, stack_effect = push_n - pop_n = -oparg-1
            assert instr.arg is not None
            return -instr.arg - 1
    return dis.stack_effect(instr.opcode, instr.arg, jump=jump)


class Space(Enum):
    locals = 1
    globals = 2
    cells = 3
    not_found = 4
