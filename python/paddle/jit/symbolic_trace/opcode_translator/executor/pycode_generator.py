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

# This class is used for abstract code generation:
# We only need to care about what type of bytecode our code needs to generate,
# without worrying about the subscripts of bytecode instructions in the code option.

from __future__ import annotations

import dis
import types

import opcode

from ...utils import (
    ResumeFnNameFactory,
    list_contain_by_id,
    list_find_index_by_id,
)
from ..instruction_utils import (
    gen_instr,
    get_instructions,
    modify_instrs,
    modify_vars,
)
from ..instruction_utils.opcode_analysis import read_write_analysis

'''
    code options for PyCodeObject
'''

pycode_attributes = [
    "co_argcount",
    "co_posonlyargcount",
    "co_kwonlyargcount",
    "co_nlocals",
    "co_stacksize",
    "co_flags",
    "co_code",
    "co_consts",
    "co_names",
    "co_varnames",
    "co_filename",
    "co_name",
    "co_firstlineno",
    "co_lnotab",
    "co_freevars",
    "co_cellvars",
]


def gen_code_options(code):
    code_options = {}
    for k in pycode_attributes:
        val = getattr(code, k)
        if isinstance(val, tuple):
            val = list(val)
        code_options[k] = val
    return code_options


'''
    generator a new code object
'''


def gen_new_opcode(instrs, code_options, keys):
    bytecode, lnotab = assemble(instrs, code_options["co_firstlineno"])
    code_options["co_lnotab"] = lnotab
    code_options["co_code"] = bytecode
    code_options["co_nlocals"] = len(code_options["co_varnames"])
    code_options["co_stacksize"] = stacksize(instrs)
    for key, val in code_options.items():
        if isinstance(val, list):
            code_options[key] = tuple(val)
    # code_options is a dict, use keys to makesure the input order
    return types.CodeType(*[code_options[k] for k in keys])


# list of instructions => bytecode & lnotab
def assemble(instructions, firstlineno):
    cur_line = firstlineno
    cur_bytecode = 0

    code = []
    lnotab = []

    for instr in instructions:
        # set lnotab
        if instr.starts_line is not None:
            line_offset = instr.starts_line - cur_line
            bytecode_offset = len(code) - cur_bytecode

            cur_line = instr.starts_line
            cur_bytecode = len(code)

            lnotab.extend(modify_lnotab(bytecode_offset, line_offset))

        # get bytecode
        arg = instr.arg or 0
        code.extend((instr.opcode, arg & 0xFF))

    return bytes(code), bytes(lnotab)


def to_byte(num):
    if num < 0:
        # -1 => 255
        num += 256
    return num


def modify_lnotab(byte_offset, line_offset):
    if byte_offset > 127:
        ret = []
        while byte_offset > 127:
            ret.extend((127, 0))
            byte_offset -= 127
        # line_offset might > 127, call recursively
        ret.extend(modify_lnotab(byte_offset, line_offset))
        return ret

    if line_offset > 127:
        # here byte_offset < 127
        ret = [byte_offset, 127]
        line_offset -= 127
        while line_offset > 0:
            ret.extend((0, line_offset))
            line_offset -= 127
        return ret

    # both < 127
    return [to_byte(byte_offset), to_byte(line_offset)]


# TODO: need to update
def stacksize(instructions):
    # two list below shows the possible stack size before opcode is called
    # the stack size might be different in different branch, so it has max and min
    max_stack = [float("-inf")] * len(instructions)
    min_stack = [float("inf")] * len(instructions)

    max_stack[0] = 0
    min_stack[0] = 0

    def update_stacksize(lasti, nexti, stack_effect):
        max_stack[nexti] = max(
            max_stack[nexti], max_stack[lasti] + stack_effect
        )
        min_stack[nexti] = min(
            min_stack[nexti], max_stack[lasti] + stack_effect
        )

    for idx in range(len(instructions)):
        instr = instructions[idx]

        if idx + 1 < len(instructions):
            stack_effect = dis.stack_effect(instr.opcode, instr.arg, jump=False)
            update_stacksize(idx, idx + 1, stack_effect)

        if instr.opcode in opcode.hasjabs or instr.opcode in opcode.hasjrel:
            stack_effect = dis.stack_effect(instr.opcode, instr.arg, jump=True)
            target_idx = instructions.index(instr.jump_to)
            update_stacksize(idx, target_idx, stack_effect)

    assert min(min_stack) >= 0
    return max(max_stack)


'''
    helper to create new code object
'''


class PyCodeGen:
    def __init__(self, frame):
        self._frame = frame
        self._origin_code = frame.f_code
        self._code_options = gen_code_options(self._origin_code)
        self._f_globals = frame.f_globals
        self._instructions = []
        self.objname_map = {}  # map from name to LOAD_GLOBAL index

    def gen_pycode(self):
        """
        return a new pycode, which is runnable.
        """
        modify_instrs(self._instructions)
        modify_vars(self._instructions, self._code_options)
        new_code = gen_new_opcode(
            self._instructions, self._code_options, pycode_attributes
        )
        return new_code

    def gen_resume_fn_at(self, index):
        self._instructions = get_instructions(self._origin_code)
        if self._instructions[index].opname == 'RETURN_VALUE':
            return None, set()
        inputs = read_write_analysis(self._instructions, index)
        self._instructions = [
            gen_instr('JUMP_ABSOLUTE', jump_to=self._instructions[index])
        ] + self._instructions

        self._code_options['co_argcount'] = len(inputs)
        # inputs should be at the front of the co_varnames
        self._code_options['co_varnames'] = tuple(
            list(inputs)
            + [
                var_name
                for var_name in self._origin_code.co_varnames
                if var_name not in inputs
            ]
        )
        fn_name = ResumeFnNameFactory().next()
        self._code_options['co_name'] = fn_name

        new_code = self.gen_pycode()
        fn = types.FunctionType(new_code, self._f_globals, fn_name)
        return fn, inputs

    def gen_loop_body_fn_between(self, start, end):
        self._instructions = get_instructions(self._origin_code)
        inputs = read_write_analysis(self._instructions, start)

        # del JUMP_ABSOLUTE at self._instructions[end-1]
        self._instructions = self._instructions[start : end - 1]
        for name in inputs:
            self.gen_load_fast(name)
        self.gen_build_tuple(len(inputs))
        self.gen_return()

        self._code_options['co_argcount'] = len(inputs)
        self._code_options['co_varnames'] = tuple(
            list(inputs)
            + [
                var_name
                for var_name in self._origin_code.co_varnames
                if var_name not in inputs
            ]
        )
        fn_name = ResumeFnNameFactory().next()
        self._code_options['co_name'] = fn_name

        new_code = self.gen_pycode()
        fn = types.FunctionType(new_code, self._f_globals, fn_name)
        return fn, inputs

    def gen_load_const(self, value):
        # Python `list.index` will find an item equal to query, i.e. `query == item`
        # returns a value of True. Since `1 == True`, this will result in an incorrect
        # index. To avoid this problem, we use id for comparison.
        if not list_contain_by_id(self._code_options["co_consts"], value):
            self._code_options["co_consts"].append(value)
        idx = list_find_index_by_id(self._code_options["co_consts"], value)
        self._add_instr("LOAD_CONST", arg=idx, argval=value)

    def gen_load_global(self, name):
        if name not in self._code_options["co_names"]:
            self._code_options["co_names"].append(name)
        idx = self._code_options["co_names"].index(name)
        self._add_instr("LOAD_GLOBAL", arg=idx, argval=name)

    def gen_load_object(self, obj, obj_name):
        if obj_name not in self.objname_map:
            self._f_globals[obj_name] = obj
            self._code_options["co_names"].append(obj_name)
            idx = len(self._code_options["co_names"]) - 1
            self.objname_map[obj_name] = idx
        idx = self.objname_map[obj_name]
        self._add_instr("LOAD_GLOBAL", arg=idx, argval=obj_name)

    def gen_store_fast(self, name):
        if name not in self._code_options["co_varnames"]:
            self._code_options["co_varnames"].append(name)
        idx = self._code_options["co_varnames"].index(name)
        self._add_instr("STORE_FAST", arg=idx, argval=name)

    def gen_load_fast(self, name):
        assert name in self._code_options["co_varnames"]
        idx = self._code_options["co_varnames"].index(name)
        self._add_instr("LOAD_FAST", arg=idx, argval=name)

    def gen_load_attr(self, name: str):
        if name not in self._code_options["co_names"]:
            self._code_options["co_names"].append(name)
        idx = self._code_options["co_names"].index(name)
        self._add_instr("LOAD_ATTR", arg=idx, argval=name)

    def gen_subscribe(self):
        self._add_instr("BINARY_SUBSCR")

    def gen_build_tuple(self, count):
        self._add_instr("BUILD_TUPLE", arg=count, argval=count)

    def gen_build_list(self, count):
        self._add_instr("BUILD_LIST", arg=count, argval=count)

    def gen_build_map(self, count):
        self._add_instr("BUILD_MAP", arg=count, argval=count)

    def gen_unpack_sequence(self, count):
        self._add_instr("UNPACK_SEQUENCE", arg=count, argval=count)

    def gen_call_function(self, argc=0):
        self._add_instr("CALL_FUNCTION", arg=argc, argval=argc)

    def gen_pop_top(self):
        self._add_instr("POP_TOP")

    def gen_return(self):
        self._add_instr("RETURN_VALUE")

    def add_pure_instructions(self, instructions):
        """
        add instructions and do nothing.
        """
        self._instructions.extend(instructions)

    def _add_instr(self, *args, **kwargs):
        instr = gen_instr(*args, **kwargs)
        self._instructions.append(instr)

    def _insert_instr(self, index, *args, **kwargs):
        instr = gen_instr(*args, **kwargs)
        self._instructions.insert(index, instr)

    def pprint(self):
        for instr in self._instructions:
            print(instr.opname, "\t\t", instr.argval)

    def gen_jump_abs(self, jump_to):
        instr = gen_instr("JUMP_ABSOLUTE", jump_to=jump_to)
        nop = gen_instr("NOP")
        self._instructions.extend([instr, nop])
        jump_to.jump_to = nop

    def extend_instrs(self, instrs):
        self._instructions.extend(instrs)
