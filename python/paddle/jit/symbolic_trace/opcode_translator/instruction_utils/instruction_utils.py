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
from typing import Any

from .opcode_info import ABS_JUMP, ALL_JUMP, REL_JUMP


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

    # for analys EXTENDED_ARG
    first_ex_arg: Instruction | None = None
    ex_arg_for: Instruction | None = None

    # used in modify_extended_args
    def __hash__(self):
        return id(self)


def gen_instr(name, arg=None, argval=None, gened=True, jump_to=None):
    return Instruction(
        opcode=dis.opmap[name],
        opname=name,
        arg=arg,
        argval=argval,
        is_generated=gened,
        jump_to=jump_to,
    )


def convert_instruction(instr):
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


def get_instructions(code):
    # instrs do not contain EXTENDED_ARG
    instrs = list(map(convert_instruction, dis.get_instructions(code)))
    for instr in instrs:
        # for 3.8, see dis.py
        if instr.opname in ALL_JUMP:
            if instr.opname in REL_JUMP:
                origin_jump_target = instr.offset + 2 + instr.arg

            elif instr.opname in ABS_JUMP:
                origin_jump_target = instr.arg

            jump_offset = origin_jump_target
            while instrs[jump_offset // 2].opname == "EXTENDED_ARG":
                jump_offset += 2

            if origin_jump_target != jump_offset:
                # copy infos from EXETENDED_ARG to other opcode
                if instrs[origin_jump_target // 2].is_jump_target:
                    instrs[jump_offset // 2].is_jump_target = instrs[
                        origin_jump_target // 2
                    ].is_jump_target
                if instrs[origin_jump_target // 2].starts_line:
                    instrs[jump_offset // 2].starts_line = instrs[
                        origin_jump_target // 2
                    ].starts_line

            instr.jump_to = instrs[jump_offset // 2]

    '''
    if the origin opcode contains EXTENDED_ARG, it should be like:
        >>  EXTENDED_ARG 1
            XX 388    <-  256 + 132
    filter all EXTENDED_ARG here
    '''
    instrs = [x for x in instrs if x.opname != "EXTENDED_ARG"]
    return instrs


'''
    modify instructions:
    1. reset offset
    2. relocate jump target
    3. add EXTENDED_ARG instruction if needed
'''


def modify_instrs(instructions):
    modify_completed = False
    while not modify_completed:
        reset_offset(instructions)
        relocate_jump_target(instructions)
        modify_completed = modify_extended_args(instructions)


def reset_offset(instructions):
    for idx, instr in enumerate(instructions):
        instr.offset = idx * 2


def relocate_jump_target(instuctions):
    extended_arg = []
    for instr in instuctions:
        if instr.opname == "EXTENDED_ARG":
            extended_arg.append(instr)
            continue

        if instr.opname in ALL_JUMP:
            # if jump target has extended_arg, should jump to the first extended_arg opcode
            jump_target = (
                instr.jump_to.offset
                if instr.jump_to.first_ex_arg is None
                else instr.jump_to.first_ex_arg.offset
            )

            if instr.opname in REL_JUMP:
                new_arg = jump_target - instr.offset - 2
            elif instr.opname in ABS_JUMP:
                new_arg = jump_target

            if extended_arg:
                instr.arg = new_arg & 0xFF
                new_arg = new_arg >> 8
                for ex in reversed(extended_arg):
                    ex.arg = new_arg & 0xFF
                    new_arg = new_arg >> 8

                # need more extended_args instr
                # set arg in the first extended_arg
                if new_arg > 0:
                    extended_arg[0].arg += new_arg << 8
            else:
                instr.arg = new_arg

        extended_arg.clear()


def modify_extended_args(instructions):
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


def modify_vars(instructions, code_options):
    co_names = code_options['co_names']
    co_varnames = code_options['co_varnames']
    for instrs in instructions:
        if instrs.opname == 'LOAD_FAST' or instrs.opname == 'STORE_FAST':
            instrs.arg = co_varnames.index(instrs.argval)
        elif instrs.opname == 'LOAD_GLOBAL':
            instrs.arg = co_names.index(instrs.argval)


'''
    utils
'''


def replace_instr(instructions, instr, new_instr):
    idx = instructions.index(instr)
    instructions[idx, idx + 1] = new_instr


def instrs_info(instrs):
    ret = []
    for idx, instr in enumerate(instrs):
        if instr.starts_line is not None:
            ret.append("")
        ret.append(
            "{line:<8s}{is_jump_target:>2s}{offset:>4d} {opname:<30s}{arg:<4s}{argval}".format(
                line=str(instr.starts_line) if instr.starts_line else "",
                is_jump_target=">>" if instr.is_jump_target else "  ",
                offset=instr.offset
                if instr.offset or instr.offset == 0
                else -1,
                opname=instr.opname,
                arg=str(instr.arg) if instr.arg else "",
                argval=f"({instr.argval})" if instr.argval else "",
            )
        )
    return "\n".join(ret)
