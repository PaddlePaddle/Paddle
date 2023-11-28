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

import random
import sys
import types
from functools import cached_property
from typing import TYPE_CHECKING

import opcode

import paddle

from ...utils import (
    FallbackError,
    InnerError,
    OrderedSet,
    ResumeFnNameFactory,
    is_clean_code,
    list_contain_by_id,
    list_find_index_by_id,
    no_eval_frame,
)
from ..instruction_utils import (
    analysis_inputs,
    apply_instr_pass,
    calc_stack_effect,
    gen_instr,
    get_instructions,
    instrs_info,
    modify_instrs,
    modify_vars,
)
from ..instruction_utils.opcode_info import (
    PYOPCODE_CACHE_SIZE,
    UNCONDITIONAL_JUMP,
    JumpDirection,
    PopJumpCond,
)
from .instr_flag import CALL_FUNCTION_EX_FLAG

CODE_NAME_RNG = random.Random(2023)

if TYPE_CHECKING:
    from typing import Any

    from ..instruction_utils import Instruction


def get_pycode_attributes() -> list[str]:
    """
    Returns a list of attribute names for PyCodeObject.
    NOTE(SigureMo): The order should consistent with signature specified in code_doc
    3.8: https://github.com/python/cpython/blob/3.8/Objects/codeobject.c#L416-L421
    3.10: https://github.com/python/cpython/blob/3.10/Objects/codeobject.c#L523-L543
    3.11: https://github.com/python/cpython/blob/3.11/Objects/codeobject.c#L1494-L1516

    Returns:
        list[str]: The attribute names for PyCodeObject.
    """
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
    ]
    if sys.version_info >= (3, 11):
        pycode_attributes.append("co_qualname")
    pycode_attributes.append("co_firstlineno")
    if sys.version_info >= (3, 10):
        pycode_attributes.append("co_linetable")
    else:
        pycode_attributes.append("co_lnotab")
    if sys.version_info >= (3, 11):
        pycode_attributes.append("co_exceptiontable")
    pycode_attributes += [
        "co_freevars",
        "co_cellvars",
    ]
    return pycode_attributes


PYCODE_ATTRIBUTES = get_pycode_attributes()


def gen_code_options(code: types.CodeType) -> dict[str, Any]:
    """
    Generates a dictionary of code options for the given code object.

    Args:
        code (types.CodeType): The code object.

    Returns:
        dict[str, any]: The code options.
    """
    code_options = {}
    for k in PYCODE_ATTRIBUTES:
        val = getattr(code, k)
        if isinstance(val, tuple):
            val = list(val)
        code_options[k] = val

    return code_options


def gen_new_opcode(
    instrs: list[Instruction], code_options: dict[str, Any], keys: list[str]
) -> types.CodeType:
    """
    Generates a new code object with the given instructions, code options, and keys.

    Args:
        instrs (list[Instruction]): The instructions for the new code object.
        code_options (dict[str, any]): The code options for the new code object.
        keys (list[str]): The keys to specify the order of code options.

    Returns:
        types.CodeType: The new code object.
    """
    bytecode, linetable = assemble(instrs, code_options["co_firstlineno"])
    if sys.version_info >= (3, 10):
        # Python deprecated co_lnotab in 3.10, use co_linetable instead
        # https://peps.python.org/pep-0626/
        code_options["co_linetable"] = linetable
    else:
        code_options["co_lnotab"] = linetable
    code_options["co_code"] = bytecode
    code_options["co_nlocals"] = len(code_options["co_varnames"])
    code_options["co_stacksize"] = stacksize(instrs)
    if sys.version_info >= (3, 11):
        # TODO: generate 3.11 exception table
        code_options["co_exceptiontable"] = bytes([])
    for key, val in code_options.items():
        if isinstance(val, list):
            code_options[key] = tuple(val)
    # code_options is a dict, use keys to makesure the input order
    return types.CodeType(*[code_options[k] for k in keys])


def assemble(
    instructions: list[Instruction], firstlineno: int
) -> tuple[bytes, bytes]:
    """
    Assembles a list of instructions into bytecode and lnotab.

    Args:
        instructions (list[Instruction]): The list of instructions to assemble.
        firstlineno (int): The starting line number.

    Returns:
        tuple[bytes, bytes]: The assembled bytecode and lnotab.
    """
    code = []
    linetable = []

    calc_linetable, update_cursor = create_linetable_calculator(firstlineno)

    for instr in instructions:
        # set linetable, Python 3.11 need to set linetable for each instruction
        if instr.starts_line is not None or sys.version_info >= (3, 11):
            linetable.extend(calc_linetable(instr.starts_line, len(code)))
            update_cursor(instr.starts_line, len(code))

        # get bytecode
        arg = instr.arg or 0
        code.extend((instr.opcode, arg & 0xFF))
        # fill CACHE
        for _ in range(get_instruction_size(instr) // 2 - 1):
            code.extend((0, 0))

    if sys.version_info >= (3, 11):
        # End hook for Python 3.11
        linetable.extend(calc_linetable(None, len(code)))
    elif sys.version_info >= (3, 10):
        # End hook for Python 3.10
        linetable.extend(calc_linetable(0, len(code)))

    return bytes(code), bytes(linetable)


def to_byte(num):
    """
    Converts a negative number to an unsigned byte.

    Args:
        num (int): The number to convert.

    Returns:
        int: The converted unsigned byte.
    """
    if num < 0:
        num += 256
    return num


def get_instruction_size(instr: Instruction) -> int:
    cache_size = 0
    if sys.version_info >= (3, 11):
        cache_size = PYOPCODE_CACHE_SIZE.get(instr.opname, 0)
    return 2 * (cache_size + 1)


def create_linetable_calculator(firstlineno: int):
    """
    Creates a line table calculator function.

    Args:
        firstlineno (int): The starting line number.

    Returns:
        Callable: The line table calculator function.
    """
    cur_lineno = firstlineno
    cur_bytecode = 0
    line_offset = 0  # For Python 3.10

    def update_cursor(starts_line: int | None, code_length: int):
        nonlocal cur_lineno, cur_bytecode
        cur_bytecode = code_length
        if starts_line is not None:
            cur_lineno = starts_line

    def calc_lnotab(starts_line: int, code_length: int):
        """
        Calculates the lnotab for Python 3.8 and 3.9.
        https://github.com/python/cpython/blob/3.9/Objects/lnotab_notes.txt

        Args:
            starts_line (int): The line number where the instruction starts.
            code_length (int): The length of the code.

        Returns:
            list[int]: The lnotab.
        """
        nonlocal cur_lineno, cur_bytecode
        line_offset = starts_line - cur_lineno
        byte_offset = code_length - cur_bytecode
        result = []

        while line_offset or byte_offset:
            line_offset_step = min(max(line_offset, -128), 127)
            byte_offset_step = min(max(byte_offset, 0), 255)
            result.extend((byte_offset_step, to_byte(line_offset_step)))
            line_offset -= line_offset_step
            byte_offset -= byte_offset_step
        return result

    def calc_linetable_py310(starts_line: int, code_length: int):
        """
        Calculates the linetable for Python 3.10.
        https://github.com/python/cpython/blob/3.10/Objects/lnotab_notes.txt

        Args:
            starts_line (int): The line number where the instruction starts.
            code_length (int): The length of the code.

        Returns:
            list[int]: The linetable.
        """
        nonlocal cur_lineno, cur_bytecode, line_offset
        byte_offset = code_length - cur_bytecode
        result = []
        while line_offset or byte_offset:
            line_offset_step = min(max(line_offset, -127), 127)
            byte_offset_step = min(max(byte_offset, 0), 254)
            result.extend((byte_offset_step, to_byte(line_offset_step)))
            line_offset -= line_offset_step
            byte_offset -= byte_offset_step
        line_offset = starts_line - cur_lineno
        return result

    def _encode_varint(num: int):
        """
        Encode unsigned integer into variable-length format.
        """
        continue_flag = 0b01 << 6
        stop_flag = 0b00 << 6
        while num >= 0x40:
            yield (num & 0x3F) | continue_flag
            num >>= 6
        yield num | stop_flag

    def _encode_svarint(num: int):
        """
        Encode signed integer into variable-length format.
        """
        unsigned_value = (((-num) << 1) | 1) if num < 0 else (num << 1)
        yield from _encode_varint(unsigned_value)

    def _encode_bytecode_to_entries_py311(line_offset: int, byte_offset: int):
        if not byte_offset:
            return []
        if 0 < byte_offset <= 8:
            entry_head = 0b1_1101_000 | (byte_offset - 1)
            return [entry_head, *list(_encode_svarint(line_offset))]
        return [
            *_encode_bytecode_to_entries_py311(line_offset, 8),
            *_encode_bytecode_to_entries_py311(line_offset, byte_offset - 8),
        ]

    def calc_linetable_py311(starts_line: int | None, code_length: int):
        """
        Calculates the linetable for Python 3.11.
        https://github.com/python/cpython/blob/3.11/Objects/locations.md

        Args:
            starts_line (int): The line number where the instruction starts.
            code_length (int): The length of the code.

        Returns:
            list[int]: The linetable.
        """
        nonlocal cur_lineno, cur_bytecode
        line_offset = starts_line - cur_lineno if starts_line is not None else 0
        byte_offset = (code_length - cur_bytecode) // 2
        return _encode_bytecode_to_entries_py311(line_offset, byte_offset)

    if sys.version_info >= (3, 11):
        return calc_linetable_py311, update_cursor
    elif sys.version_info >= (3, 10):
        return calc_linetable_py310, update_cursor
    else:
        return calc_lnotab, update_cursor


def compile_exception_table():
    """Compile the exception table, it is used for Python 3.11+.
    See https://github.com/python/cpython/blob/3.11/Objects/exception_handling_notes.txt
    """
    # TODO
    ...


def stacksize(instructions: list[Instruction]) -> float:
    """
    Calculates the maximum stack size before each opcode is called.

    Args:
        instructions (list[Instruction]): The list of instructions.

    Returns:
        int: The maximum stack size.
    """
    max_stack = [float("-inf")] * len(instructions)

    max_stack[0] = 0

    queue = []
    queue.append(0)

    def update_stacksize(lasti: int, nexti: int, stack_effect: int):
        """
        Updates the maximum stack size.

        Args:
            lasti (int): The index of the last instruction.
            nexti (int): The index of the next instruction.
            stack_effect (int): The effect on the stack size.

        Returns:
            None
        """
        old_max = max_stack[nexti]
        max_stack[nexti] = max(
            max_stack[nexti], max_stack[lasti] + stack_effect
        )
        if old_max != max_stack[nexti]:
            if nexti not in queue:  # may be slow, we can use a flag.
                queue.append(nexti)

    while len(queue) > 0:
        idx = queue[0]
        del queue[0]
        instr = instructions[idx]
        opname = instr.opname
        if (
            idx + 1 < len(instructions)
            and instr.opname not in UNCONDITIONAL_JUMP
        ):
            stack_effect = calc_stack_effect(instr, jump=False)
            update_stacksize(idx, idx + 1, stack_effect)

        if instr.opcode in opcode.hasjabs or instr.opcode in opcode.hasjrel:
            stack_effect = calc_stack_effect(instr, jump=True)
            target_idx = instructions.index(instr.jump_to)
            update_stacksize(idx, target_idx, stack_effect)

    # assert min(min_stack) >= 0 # min_stack may be a negative number when try: except is got.
    return max(max_stack)


class PyCodeGen:
    """Helper to create new code object"""

    def __init__(
        self, frame: types.FrameType, disable_eval_frame: bool = False
    ):
        """
        Initializes a PyCodeGen object.

        Args:
            frame: The frame to be translated.
            disable_eval_frame (bool): Whether to disable the evaluation frame. Defaults to False.
        """
        self._frame = frame
        self._origin_code = frame.f_code
        self._code_options = gen_code_options(self._origin_code)
        self.update_code_name("", is_resumed_fn=False)
        self._f_globals = frame.f_globals
        self._instructions = []
        self.disable_eval_frame = disable_eval_frame
        self.hooks = []
        if self.disable_eval_frame:
            self.gen_disable_eval_frame()

    def insert_prefix_instructions(self):
        """
        Insert prefix instructions to the instruction list.
        In Python 3.11+, we need to insert MAKE_CELL and COPY_FREE_VARS before the
        first instruction.
        The implementation is based on cpython implementation:
        https://github.com/python/cpython/blob/f45ef5edabb1cc0748f3326e7114b8aaa0424392/Python/compile.c#L8177
        """
        prefixes = []
        if sys.version_info >= (3, 11):
            if self._code_options["co_cellvars"]:
                # Insert MAKE_CELL
                name_map = list(
                    OrderedSet(self._code_options["co_varnames"])
                    | OrderedSet(self._code_options["co_cellvars"])
                )

                for i in self._code_options["co_cellvars"]:
                    idx: int = name_map.index(i)
                    prefixes.append(gen_instr("MAKE_CELL", arg=idx, argval=i))

            if self._code_options["co_freevars"]:
                n_freevars = len(self._code_options["co_freevars"])
                # Insert COPY_FREE_VARS
                prefixes.append(
                    gen_instr(
                        "COPY_FREE_VARS", arg=n_freevars, argval=n_freevars
                    )
                )

            # Insert RESUME
            prefixes.append(gen_instr("RESUME", arg=0, argval=0))
        self._instructions[:] = prefixes + self._instructions

    def update_code_name(self, fn_name, is_resumed_fn):
        if is_resumed_fn:
            self._code_options[
                'co_name'
            ] = f"${fn_name}@{self._code_options['co_name'][1:]}"
        else:
            if self._code_options['co_name'].startswith("$"):
                self._code_options[
                    'co_name'
                ] = f"#{self._code_options['co_name']}"
            elif not self._code_options['co_name'].startswith("#"):
                random_number = int(CODE_NAME_RNG.random() * 100000000)
                self._code_options[
                    'co_name'
                ] = f"#{self._code_options['co_name']}_{hex(random_number & 0xFFFFF)[2:]:0>5}"

    def gen_pycode(self) -> types.CodeType:
        """
        Generates a new pycode that is runnable.

        Returns:
            CodeType: The generated code object.
        """
        for hook in self.hooks:
            hook()
        self.hooks.clear()

        self.insert_prefix_instructions()
        apply_instr_pass(self._instructions, self._code_options)
        modify_instrs(self._instructions)
        modify_vars(self._instructions, self._code_options)
        new_code = gen_new_opcode(
            self._instructions, self._code_options, PYCODE_ATTRIBUTES
        )

        return new_code

    def gen_resume_fn_at(
        self, index: int, stack_size: int
    ) -> tuple[None | types.FunctionType, OrderedSet[str]]:
        """
        Generates a resume function at the specified index in the instruction list.

        Args:
            index (int): The index in the instruction list to generate the resume function.
            stack_size (int): The size of the stack. Defaults to 0.

        Returns:
            tuple: The resume function object and the inputs to the function.

        """

        self._instructions = get_instructions(self._origin_code)
        # TODO(dev): could give an example code here?
        if self._instructions[index].opname == 'RETURN_VALUE':
            return None, OrderedSet()
        inputs = analysis_inputs(self._instructions, index)
        fn_name = ResumeFnNameFactory().next()
        stack_arg_str = fn_name + '_stack_{}'

        self._instructions = (
            [
                gen_instr('LOAD_FAST', argval=stack_arg_str.format(i))
                for i in range(stack_size)
            ]
            + [gen_instr('JUMP_FORWARD', jump_to=self._instructions[index])]
            + self._instructions
        )

        self._code_options['co_argcount'] = len(inputs) + stack_size
        # inputs should be at the front of the co_varnames
        self._code_options['co_varnames'] = list(
            [stack_arg_str.format(i) for i in range(stack_size)]
            + list(inputs)
            + [
                var_name
                for var_name in self._code_options['co_varnames']
                if var_name not in inputs
            ]
        )

        self.update_code_name(fn_name, is_resumed_fn=True)
        new_code = self.gen_pycode()
        if len(new_code.co_freevars) + len(new_code.co_cellvars) > 0:
            raise FallbackError("Break graph in closure is not support.")
        fn = types.FunctionType(new_code, self._f_globals, new_code.co_name)

        return fn, inputs

    @cached_property
    def global_null_variable(self):
        from .variables.basic import NullVariable

        return NullVariable()

    def gen_disable_eval_frame(self):
        """
        Generates instructions to disable the evaluation frame.
        """
        if is_clean_code():
            return
        self.gen_load_object(
            paddle.framework.core.set_eval_frame, "paddle_set_eval_frame_fn"
        )
        self.gen_load_const(None)
        self.gen_call_function(1)
        self.gen_store_fast("___old_eval_frame")

    def gen_enable_eval_frame(self):
        """
        Generates instructions to enable the evaluation frame.
        """
        if is_clean_code():
            return
        self.gen_load_object(
            paddle.framework.core.set_eval_frame, "paddle_set_eval_frame_fn"
        )
        self.gen_load_fast("___old_eval_frame")
        self.gen_call_function(1)
        self.gen_pop_top()

    def gen_outputs_and_return(self, outputs):
        for name in outputs:
            self.gen_load(name)
        self.gen_build_tuple(len(outputs))
        self.gen_return()

    def create_fn_with_inputs(self, inputs: list) -> types.FunctionType:
        """
        Creates a function with specific input and output variables.

        Args:
            inputs (list): The input variables.

        Returns:
            function: The created function object.
        """
        self._code_options['co_argcount'] = len(inputs)
        self._code_options['co_varnames'] = list(
            list(inputs)
            + [
                var_name
                for var_name in self._origin_code.co_varnames
                if var_name not in inputs
            ]
        )
        fn_name = ResumeFnNameFactory().next()
        self.update_code_name(fn_name, is_resumed_fn=True)
        new_code = self.gen_pycode()
        if len(new_code.co_freevars) + len(new_code.co_cellvars) > 0:
            raise FallbackError("Break graph in closure is not support.")
        fn = types.FunctionType(new_code, self._f_globals, new_code.co_name)
        return fn

    def gen_load_const(self, value: Any):
        """
        Generates instructions to load a constant value.
        """
        # Python `list.index` will find an item equal to query, i.e. `query == item`
        # returns a value of True. Since `1 == True`, this will result in an incorrect
        # index. To avoid this problem, we use id for comparison.
        if not list_contain_by_id(self._code_options["co_consts"], value):
            self._code_options["co_consts"].append(value)
        idx = list_find_index_by_id(self._code_options["co_consts"], value)
        self._add_instr("LOAD_CONST", arg=idx, argval=value)

    def gen_print_log(self, message):
        """print a log"""
        import paddle

        self.gen_load_object(
            paddle.framework.core.set_eval_frame, "dbg_set_eval_frame"
        )
        self.gen_load_const(None)
        self.gen_call_function(1)
        self.gen_store_fast("old_eval_frame")
        self.gen_load_global("print", push_null=True)
        self.gen_load_const(message)
        self.gen_call_function(1)
        self.gen_pop_top()
        self.gen_load_object(
            paddle.framework.core.set_eval_frame, "dbg_set_eval_frame"
        )
        self.gen_load_fast("old_eval_frame")
        self.gen_call_function(1)
        self.gen_pop_top()

    def gen_dbg_function(self, dbg_fun):
        """debug bytecode helper function.
        Usage like:
        def dbg_func():
            import inspect
            import dis
            print("dbg here.")
            print(locals())
            frame = inspect.currentframe().f_back
            code = (inspect.currentframe().f_back.f_code)
            breakpoint()
            print(inspect.currentframe().f_back.f_locals['y'])

        self.pycode_gen.gen_dbg_function(dbg_func)
        """
        import paddle

        self.gen_load_object(
            paddle.framework.core.set_eval_frame, "dbg_set_eval_frame"
        )
        self.gen_load_const(None)
        self.gen_call_function(1)
        self.gen_store_fast("old_eval_frame")
        self.gen_load_object(dbg_fun, "dbg1")
        self.gen_call_function(0)
        self.gen_pop_top()
        self.gen_load_object(
            paddle.framework.core.set_eval_frame, "dbg_set_eval_frame"
        )
        self.gen_load_fast("old_eval_frame")
        self.gen_call_function(1)
        self.gen_pop_top()

    @property
    def cell_free_storage(self):
        return (
            self._code_options["co_cellvars"]
            + self._code_options["co_freevars"]
        )

    def gen_load(self, name):
        if name in self.cell_free_storage:
            self.gen_load_deref(name)
        elif name in self._code_options["co_varnames"]:
            self.gen_load_fast(name)
        elif name in self._code_options["co_names"]:
            self.gen_load_global(name, push_null=False)
        else:
            raise InnerError(
                f"Want gen_load, but {name} can not found in code object."
            )

    def gen_store(self, name, code):
        """
        Generate the bytecode for storing a variable identified by 'name'
        in the corresponding symbol table and generate the appropriate
        store code based on the symbol table analysis.

        Args:
            name (str): The name of the variable.
        """
        if name in (code.co_freevars + code.co_cellvars):
            self.gen_store_deref(name)
        elif name in code.co_varnames:
            self.gen_store_fast(name)
        elif name in code.co_names:
            self.gen_store_global(name)
        else:
            raise InnerError(
                f"Want gen_store, but {name} can not found in code object."
            )

    def gen_load_global(self, name, push_null=False):
        """
        Generate the bytecode for loading a global variable.

        Args:
            name (str): The name of the global variable.
        """
        if name not in self._code_options["co_names"]:
            self._code_options["co_names"].append(name)
        idx = self._code_options["co_names"].index(name)
        if sys.version_info >= (3, 11):
            idx <<= 1
            if push_null:
                idx |= 1
        self._add_instr("LOAD_GLOBAL", arg=idx, argval=name)

    def gen_load_object(self, obj, obj_name: str, push_null: bool = True):
        """
        Generate the bytecode for loading an object.

        Args:
            obj (Any): The object to load.
            obj_name (str): The name of the object.
        """

        if obj_name not in self._f_globals:
            self._f_globals[obj_name] = obj
        self.gen_load_global(obj_name, push_null=push_null)

    def gen_load_null_variable(self):
        """
        Generate the bytecode for loading a null variable.
        """
        null_var = self.global_null_variable
        self.gen_load_object(null_var, "___null_var", push_null=False)

    def gen_load_fast(self, name):
        """
        Generate the bytecode for loading a local variable.

        Args:
            name (str): The name of the local variable.
        """
        if name not in self._code_options["co_varnames"]:
            self._code_options["co_varnames"].append(name)
        idx = self._code_options["co_varnames"].index(name)
        self._add_instr("LOAD_FAST", arg=idx, argval=name)

    def gen_load_deref(self, name):
        if name not in self.cell_free_storage:
            self._code_options["co_freevars"].append(name)
        if sys.version_info >= (3, 11):
            # Because the co_varnames maybe changed after other codegen
            # operations, we need re-calculate the index in modify_vars
            idx = (
                self._code_options["co_varnames"]
                + self._code_options["co_freevars"]
            ).index(name)
        else:
            idx = self.cell_free_storage.index(name)
        self._add_instr("LOAD_DEREF", arg=idx, argval=name)

    def gen_load_attr(self, name: str):
        if name not in self._code_options["co_names"]:
            self._code_options["co_names"].append(name)
        idx = self._code_options["co_names"].index(name)
        self._add_instr("LOAD_ATTR", arg=idx, argval=name)

    def gen_store_attr(self, name: str):
        if name not in self._code_options["co_names"]:
            self._code_options["co_names"].append(name)
        idx = self._code_options["co_names"].index(name)
        self._add_instr("STORE_ATTR", arg=idx, argval=name)

    def gen_delete_attr(self, name: str):
        if name not in self._code_options["co_names"]:
            self._code_options["co_names"].append(name)
        idx = self._code_options["co_names"].index(name)
        self._add_instr("DELETE_ATTR", arg=idx, argval=name)

    def gen_load_method(self, name: str):
        if name not in self._code_options["co_names"]:
            self._code_options["co_names"].append(name)
        idx = self._code_options["co_names"].index(name)
        self._add_instr("LOAD_METHOD", arg=idx, argval=name)

    def gen_delete_global(self, name: str):
        if name not in self._code_options["co_names"]:
            self._code_options["co_names"].append(name)
        idx = self._code_options["co_names"].index(name)
        self._add_instr("DELETE_GLOBAL", arg=idx, argval=name)

    def gen_import_name(self, name: str):
        if name not in self._code_options["co_names"]:
            self._code_options["co_names"].append(name)
        idx = self._code_options["co_names"].index(name)
        self._add_instr("IMPORT_NAME", arg=idx, argval=name)

    def gen_store_fast(self, name):
        if name not in self._code_options["co_varnames"]:
            self._code_options["co_varnames"].append(name)
        idx = self._code_options["co_varnames"].index(name)
        self._add_instr("STORE_FAST", arg=idx, argval=name)

    def gen_store_global(self, name):
        if name not in self._code_options["co_names"]:
            self._code_options["co_names"].append(name)
        idx = self._code_options["co_names"].index(name)
        self._add_instr("STORE_GLOBAL", arg=idx, argval=name)

    def gen_store_deref(self, name):
        if name not in self.cell_free_storage:
            self._code_options["co_freevars"].append(name)
        if sys.version_info >= (3, 11):
            # Because the co_varnames maybe changed after other codegen
            # operations, we need re-calculate the index in modify_vars
            idx = (
                self._code_options["co_varnames"]
                + self._code_options["co_freevars"]
            ).index(name)
        else:
            idx = self.cell_free_storage.index(name)
        self._add_instr("STORE_DEREF", arg=idx, argval=name)

    def gen_store_subscr(self):
        self._add_instr("STORE_SUBSCR")

    def gen_subscribe(self):
        self._add_instr("BINARY_SUBSCR")

    def gen_build_tuple(self, count):
        self._add_instr("BUILD_TUPLE", arg=count, argval=count)

    def gen_build_list(self, count):
        self._add_instr("BUILD_LIST", arg=count, argval=count)

    def gen_build_map(self, count):
        self._add_instr("BUILD_MAP", arg=count, argval=count)

    def gen_build_slice(self, argc):
        self._add_instr("BUILD_SLICE", arg=argc, argval=argc)

    def gen_unpack_sequence(self, count):
        self._add_instr("UNPACK_SEQUENCE", arg=count, argval=count)

    def gen_call_function(self, argc=0):
        if sys.version_info >= (3, 11):
            self._add_instr("PRECALL", arg=argc, argval=argc)
            self._add_instr("CALL", arg=argc, argval=argc)
        else:
            self._add_instr("CALL_FUNCTION", arg=argc, argval=argc)

    def gen_call_function_ex(self, has_kwargs):
        flag = 0
        if has_kwargs:
            flag |= CALL_FUNCTION_EX_FLAG.CFE_HAS_KWARGS
        self._add_instr("CALL_FUNCTION_EX", arg=flag, argval=flag)

    def gen_call_method(self, argc=0):
        if sys.version_info >= (3, 11):
            self._add_instr("PRECALL", arg=argc, argval=argc)
            self._add_instr("CALL", arg=argc, argval=argc)
        else:
            self._add_instr("CALL_METHOD", arg=argc, argval=argc)

    def gen_kw_names(self, kw_names: tuple[str, ...] | None):
        if kw_names is None:
            return
        if sys.version_info < (3, 11):
            raise InnerError("gen_kw_names is not supported before python3.11")
        if kw_names not in self._code_options["co_consts"]:
            self._code_options["co_consts"].append(kw_names)
        idx = self._code_options["co_consts"].index(kw_names)
        self._add_instr("KW_NAMES", arg=idx, argval=kw_names)

    def gen_pop_top(self):
        self._add_instr("POP_TOP")

    def gen_rot_n(self, n):
        if n <= 1:
            return
        if sys.version_info >= (3, 11):
            for i in range(n, 1, -1):
                self._add_instr("SWAP", arg=i)
        elif sys.version_info >= (3, 10):
            self._add_instr("ROT_N", arg=n)
        else:
            if n <= 4:
                self._add_instr("ROT_" + ["TWO", "THREE", "FOUR"][n - 2])
            else:

                def rot_n_fn(n):
                    vars = [f"var{i}" for i in range(n)]
                    rotated = reversed(vars[-1:] + vars[:-1])
                    fn = eval(f"lambda {','.join(vars)}: ({','.join(rotated)})")
                    fn = no_eval_frame(fn)
                    fn.__name__ = f"rot_{n}_fn"
                    return fn

                self.gen_build_tuple(n)
                self.gen_load_const(rot_n_fn(n))
                self.gen_rot_n(2)
                self._add_instr("CALL_FUNCTION_EX", arg=0)
                self.gen_unpack_sequence(n)

    def gen_shift_n(self, s: int, n: int):
        """
        Generate the bytecode for shifting the stack.

        Args:
            s (int): Steps to shift.
            n (int): The number of elements to shift.
        """
        if s == 0 or n <= 1:
            return

        # NOTE(zrr1999): right shift s steps is equal to left shift n-s steps
        if abs(s) > n // 2:
            new_s = s - n if s > 0 else s + n
            self.gen_shift_n(new_s, n)
            return
        if s > 0:
            # NOTE: s=1, n=3 [1,2,3,4,5] -> [1,2,5,3,4]
            #       s=2, n=3 [1,2,3,4,5] -> [1,2,4,5,3]
            if s == 1:
                self.gen_rot_n(n)
            else:
                self.gen_rot_n(n)
                self.gen_shift_n(s - 1, n)

        else:  # s < 0
            if sys.version_info >= (3, 11):
                # NOTE: s=-1, n=3 [1,2,3,4,5] -> [1,2,4,5,3]
                if s == -1:
                    for i in range(2, n + 1):
                        self._add_instr("SWAP", arg=i)
                else:
                    self.gen_shift_n(-1, n)
                    self.gen_shift_n(s + 1, n)
            else:
                raise NotImplementedError(
                    "shift_n is not supported before python3.11"
                )

    def gen_swap(self, n):
        if sys.version_info >= (3, 11):
            self._add_instr("SWAP", arg=n)
        else:
            raise NotImplementedError("swap is not supported before python3.11")

    def gen_jump(
        self,
        jump_to: Instruction | None = None,
        *,
        direction: JumpDirection = JumpDirection.FORWARD,
    ) -> Instruction:
        if sys.version_info >= (3, 11):
            return self._add_instr(f"JUMP_{direction.value}", jump_to=jump_to)
        else:
            return self._add_instr("JUMP_ABSOLUTE", jump_to=jump_to)

    def gen_pop_jump(
        self,
        jump_to: Instruction | None = None,
        *,
        direction: JumpDirection = JumpDirection.FORWARD,
        suffix: PopJumpCond = PopJumpCond.NONE,
    ) -> Instruction:
        if sys.version_info >= (3, 11):
            return self._add_instr(
                f"POP_JUMP_{direction.value}_IF_{suffix.value}", jump_to=jump_to
            )
        else:
            return self._add_instr(
                f"POP_JUMP_IF_{suffix.value}", jump_to=jump_to
            )

    def gen_return(self):
        self._add_instr("RETURN_VALUE")

    def gen_get_iter(self):
        self._add_instr("GET_ITER")

    def gen_operator_only(self, op_name):
        """
        only generator operator instruction, do nothing for
        operands.
        """
        self._add_instr(op_name)

    def gen_operator(self, op_name):
        """
        only generator operator instruction, do nothing for
        operands.
        """
        self._add_instr(op_name)

    def gen_compare(self, cmp_op):
        """
        only generator operator instruction, do nothing for
        operands.
        """
        self._add_instr("COMPARE_OP", cmp_op)

    def _add_instr(self, *args, **kwargs):
        instr = gen_instr(*args, **kwargs)
        self._instructions.append(instr)
        return instr

    def _insert_instr(self, index, *args, **kwargs):
        instr = gen_instr(*args, **kwargs)
        self._instructions.insert(index, instr)

    def pprint(self):
        print(instrs_info(self._instructions))

    def extend_instrs(self, instrs):
        self._instructions.extend(instrs)

    def pop_instr(self):
        self._instructions.pop()
