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
import functools
import inspect
import operator
import sys
import traceback
import types
from dataclasses import dataclass
from itertools import chain
from typing import Any, Callable

import opcode

from ...profiler import EventGuard, event_register
from ...psdb import NO_BREAKGRAPH_CODES
from ...utils import (
    ENV_MIN_GRAPH_SIZE,
    BreakGraphError,
    FallbackError,
    InnerError,
    OrderedSet,
    SotUndefinedVar,
    log,
    log_do,
)
from ..custom_code import CustomCode
from ..instruction_utils import (
    Instruction,
    Space,
    analysis_inputs,
    analysis_used_names_with_space,
    calc_stack_effect,
    get_instructions,
)
from ..instruction_utils.opcode_info import JumpDirection, PopJumpCond
from .dispatch_functions import (
    operator_BAD,
    operator_exception_match,
    operator_in,
    operator_is_none,
    operator_is_not_none,
    operator_not_in,
)
from .dispatcher import Dispatcher
from .function_graph import FunctionGraph
from .instr_flag import CALL_FUNCTION_EX_FLAG as CFE
from .instr_flag import FORMAT_VALUE_FLAG as FV
from .instr_flag import MAKE_FUNCTION_FLAG as MF
from .pycode_generator import PyCodeGen
from .tracker import (
    CellTracker,
    ConstTracker,
    DanglingTracker,
    DummyTracker,
    LocalTracker,
)
from .variable_stack import VariableStack
from .variables import (
    BuiltinVariable,
    CellVariable,
    ConstantVariable,
    ContainerVariable,
    DictVariable,
    GlobalVariable,
    ListVariable,
    MethodVariable,
    NullVariable,
    SequenceIterVariable,
    SliceVariable,
    TensorVariable,
    TupleVariable,
    UserDefinedFunctionVariable,
    VariableBase,
    VariableFactory,
)

SUPPORT_COMPARE_OP = {
    ">": operator.gt,
    "<": operator.lt,
    ">=": operator.ge,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
    "is not": operator.is_not,
    "is": operator.is_,
    "in": operator_in,
    "not in": operator_not_in,
    "exception match": operator_exception_match,
    "BAD": operator_BAD,
}


@dataclass
class Stop:
    state: str


def tos_op_wrapper(fn: Callable):
    """
    A decorator function that wraps an opcode operation and applies certain functionality to it.

    Args:
        fn: The opcode operation to be wrapped.

    Returns:
        The wrapped opcode operation.
    """
    nargs = len(inspect.signature(fn).parameters)

    @call_break_graph_decorator(push_n=1)
    def inner(self: OpcodeExecutorBase, instr: Instruction):
        args = self.stack.pop_n(nargs)
        res = BuiltinVariable(fn, graph=self._graph, tracker=DanglingTracker())(
            *args
        )
        self.stack.push(res)

    return inner


def tos_inplace_op_wrapper(fn: Callable):
    """
    A decorator function that wraps an inplace opcode operation and applies certain functionality to it.

    Args:
        fn: The inplace opcode operation to be wrapped.

    Returns:
        The wrapped inplace opcode operation.

    """

    @call_break_graph_decorator(push_n=1)
    def inner(self: OpcodeExecutorBase, instr: Instruction):
        """
        Inner function that represents the wrapped inplace opcode operation.

        Args:
            self: The instance of the OpcodeExecutorBase class.
            instr: The instruction to be executed.

        """
        args = self.stack.pop_n(2)
        res = BuiltinVariable(fn, graph=self._graph, tracker=DanglingTracker())(
            *args
        )
        res.debug_name = args[0].debug_name
        self.stack.push(res)

    return inner


def pop_jump_if_op_wrapper(fns: list[Callable[[Any], Any]]):
    """
    A decorator function that wraps a POP_JUMP_*_IF_* opcode operation and applies certain functionality to it.

    Args:
        fn: The condition function.

    Returns:
        The wrapped POP_JUMP_*_IF_* opcode operation.

    """

    @jump_break_graph_decorator
    def inner(self: OpcodeExecutorBase, instr: Instruction):
        """
        Inner function that represents the wrapped POP_JUMP_IF opcode operation.

        Args:
            self: The instance of the OpcodeExecutorBase class.
            instr: The instruction to be executed.

        """
        pred_obj = self.stack.pop()

        try:
            self._graph.add_global_guarded_variable(pred_obj)
            res = pred_obj
            for fn in fns:
                res = BuiltinVariable(
                    fn, graph=self._graph, tracker=DanglingTracker()
                )(res)

            assert isinstance(res, ConstantVariable)
            is_jump = res.get_py_value()
            assert isinstance(is_jump, bool)
            if is_jump:
                assert instr.jump_to is not None
                self.jump_to(instr.jump_to)
        except BreakGraphError:
            raise FallbackError(
                f"Currently don't support predicate {pred_obj.__class__.__name__}"
            )

    return inner


def jump_break_graph_decorator(normal_jump: Callable):
    """
    A decorator function that breaks off the graph when a JUMP-related instruction is encountered.

    Args:
        normal_jump: The normal jump operation.

    Returns:
        The wrapped jump operation.

    """

    def inner(self: OpcodeExecutor, instr: Instruction):
        result = self.stack.top
        if isinstance(result, TensorVariable):
            self.stack.pop()
            # fallback when in OpcodeExecutor
            # raise error in OpcodeInlineExecutor
            log(3, "[BreakGraph] jump break graph, because if tensor\n")
            self._break_graph_in_jump(result, instr)
            return Stop(state="BreakGraph")
        else:
            return normal_jump(self, instr)

    return inner


def call_break_graph_decorator(push_n: int | Callable[[int | None], int]):
    """
    A decorator function that breaks off the graph when a function CALL instruction is encountered.

    Args:
        push_n: The number of arguments to be pushed onto the stack.

    Returns:
        The decorated function.

    """

    def decorate(call_fn: Callable):
        @functools.wraps(call_fn)
        def wrapper(self: OpcodeExecutor, instr: Instruction):
            origin_stack = self.stack.copy()
            try:
                return call_fn(self, instr)
            except BreakGraphError as e:
                if self._code in NO_BREAKGRAPH_CODES:
                    raise InnerError(
                        f"{self._code.co_name} should not break graph, but got '{e}'"
                    )
                if isinstance(self, OpcodeExecutor):
                    log(3, f"[BreakGraph] call function Break graph: {e}\n")
                    self._break_graph_in_call(origin_stack, instr, push_n)
                    return Stop(state="BreakGraph")
                else:
                    raise e

        return wrapper

    return decorate


def fallback_when_occur_error(fn: Callable):
    """
    A decorator function that provides fallback behavior when an error occurs during graph processing.

    Args:
        fn: The function to be wrapped.

    Returns:
        The wrapped function.

    """

    def inner(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            raise FallbackError(
                f'[Fallback] An exception occurred when processing break graph, fallback to dygraph, error message is: \n{type(e)} : {e}\n'
            )

    return inner


class OpcodeExecutorBase:
    """
    Base class for executing opcode instructions.

    The OpcodeExecutorBase class provides methods and functionality to execute opcode instructions.

    If you want to learn more about Python instructions, see https://docs.python.org/3/library/dis.html for details.

    Args:
        code: The bytecode of the function to be executed.
        graph: The function graph.

    Attributes:
        call_stack (list[OpcodeExecutorBase]): A list to keep track of the call stack.
        _stack (list[VariableBase]): The stack used for storing variables during execution.
        _co_consts: List to store constants.
        _locals (dict): Dictionary to store local variables.
        _globals (dict): Dictionary to store global variables.
        _builtins (dict): Dictionary to store built-in variables.
        _lasti (int): Index of the last executed instruction.
        _code (types.CodeType): The code object to be executed.
        _instructions: Iterator of opcode instructions.
        _graph (FunctionGraph): The function graph representing the code.
        _current_line: The current line number of the execution.
        new_code: Placeholder for new code (to be generated by PyCodeGen).
        guard_fn: Placeholder for guard function.
        _name (str): Name of the executor.

    """

    call_stack: list[OpcodeExecutorBase] = []

    @staticmethod
    def validate_value(value):
        assert isinstance(
            value, VariableBase
        ), f"value: {value}, type shoule be VariableBase(or derived), but get {type(value)}"
        assert not isinstance(value.tracker, DanglingTracker) or isinstance(
            value, (NullVariable, CellVariable)
        ), f"dangling variable {value} should not be pushed into stack."

    def __init__(self, code: types.CodeType, graph: FunctionGraph):
        OpcodeExecutorBase.call_stack.append(self)
        # fake env for run, new env should be gened by PyCodeGen
        self.stack = VariableStack(validate_value_func=self.validate_value)
        self._co_consts = []
        self._locals = {}
        self._globals: GlobalVariable = None  # type: ignore
        self._builtins = {}
        self._cells = {}  # position to put cells
        self._lasti = 0  # idx of instruction list
        self._code = code
        self._current_line: int = -1
        self._instructions = get_instructions(self._code)
        self._graph = graph
        self.new_code: types.CodeType | None = None
        self.guard_fn = None
        self._name = "Executor"
        self._call_shape: tuple[
            str, ...
        ] | None = None  # store kwnames for Python 3.11+
        self._prepare_virtual_env()

        self.stop_state = None

    def print_sir(self):
        """
        Prints the Static Instruction Representation (SIR) in the executor.

        """
        print(self._graph.sir_ctx.TOS)

    def _prepare_virtual_env(self):
        """
        Prepares the virtual environment for the executor.

        Raises:
            NotImplementedError: If the method is not implemented.

        """
        raise NotImplementedError("Please implement virtual_env.")

    def _break_graph_in_jump(self, result, instr: Instruction):
        """
        Breaks the graph in JUMP instructions.

        Args:
            result: The execution result.
            instr: The jump instruction.

        Raises:
            NotImplementedError: If the method is not implemented.

        """
        raise NotImplementedError()

    def transform(self):
        """
        Abstract method need to be implemented to symbolic translate each instruction.

        Raises:
            NotImplementedError: If the method is not implemented.

        """
        raise NotImplementedError()

    def get_var(self, name: str):
        """
        Gets the variable with the given name.

        Args:
            name: The name of the variable.

        Returns:
            The variable.

        Raises:
            InnerError: If the variable cannot be found.

        """
        if name in self._locals.keys():
            return self._locals[name]
        elif name in self._cells.keys():  # in closure
            return self._cells[name].cell_content()
        elif name in self._globals.keys():
            return self._globals.get(name)
        elif name in self._builtins.keys():
            return self._builtins[name]
        else:
            raise InnerError(f'Can not get var: {name}')

    def has_var(self, name: str, space: str = "any"):
        if space == "any":
            return name in set(
                chain(
                    self._locals.keys(),
                    self._cells.keys(),
                    self._globals.keys(),
                    self._builtins.keys(),
                )
            )
        elif space == Space.locals:
            return name in self._locals
        elif space == Space.cells:
            return name in self._cells
        elif space == Space.globals:
            return name in set(
                chain(
                    self._globals.keys(),
                    self._builtins.keys(),
                )
            )
        return False

    def pop_call_stack_until_self(self):
        """
        Pops the call stack until the current executor.

        """
        assert (
            self in OpcodeExecutorBase.call_stack
        ), f"{self} not in call stack"
        while OpcodeExecutorBase.call_stack.pop() is not self:
            pass

    @staticmethod
    def error_message_summary(original_error: Exception) -> str:
        """
        Creates a summary of the error message during execution.

        Args:
            original_error: The original error.

        Returns:
            The summary error message.

        """
        indent = 2 * " "
        message_lines = ["In simulate execution:", ""]
        for current_simulator in OpcodeExecutorBase.call_stack:
            code = current_simulator._code
            current_line = current_simulator._current_line
            lines, start = inspect.getsourcelines(code)
            real_name = code.co_name
            message_lines.append(
                f"{indent}  File \"{code.co_filename}\", line {current_line}, in {real_name}"
            )
            if current_line != -1:
                message_lines.append(
                    f"{indent}  {lines[current_line-start].rstrip()}"
                )
        error_message = traceback.format_exception_only(
            type(original_error), original_error
        )
        for line in error_message:
            line = line.rstrip()
            message_lines.append(f"{indent}  {line}")
        return "\n".join(message_lines)

    def run(self):
        """
        Executes the opcode.

        """
        log(3, f"start execute opcode: {self._code}\n")
        self._lasti = 0
        while True:
            if self._lasti >= len(self._instructions):
                raise InnerError("lasti out of range, InnerError.")
            cur_instr = self._instructions[self._lasti]
            self._lasti += 1
            is_stop = self.step(cur_instr)
            if is_stop:
                self.stop_state = is_stop.state
                self.pop_call_stack_until_self()
                break

    def step(self, instr: Instruction):
        """
        Executes a single step of the opcode.

        Args:
            instr: The instruction to be executed.

        Returns:
            True if execution should stop, False otherwise.

        Raises:
            FallbackError: If the opcode is not supported.

        """
        if instr.starts_line is not None:
            self._current_line = instr.starts_line
        if not hasattr(self, instr.opname):
            raise FallbackError(f"opcode: {instr.opname} is not supported.")
        log_message = f"[Translate {self._name}]: (line {self._current_line:>3}) {instr.opname:<12} {instr.argval}, stack is {self.stack}\n"
        log(3, log_message)
        code_file = self._code.co_filename
        code_line = self._current_line
        code_name = self._code.co_name
        code_offset = instr.offset
        from ..breakpoint import BreakpointManager

        if BreakpointManager().hit(
            code_file, code_line, code_name, code_offset
        ):
            BreakpointManager().locate(self)
            print(log_message)
            breakpoint()  # breakpoint for debug

        with EventGuard(f"{instr.opname}", event_level=1):
            return getattr(self, instr.opname)(instr)  # run single step.

    def indexof(self, instr: Instruction):
        """
        Gets the index of the instruction.

        Args:
            instr: The instruction.

        Returns:
            The index of the instruction.

        """
        return self._instructions.index(instr)

    def jump_to(self, instr: Instruction):
        """
        Jumps to the given instruction.

        Args:
            instr: The instruction to jump to.

        """
        self._lasti = self.indexof(instr)

    def COPY(self, instr: Instruction):
        assert isinstance(instr.arg, int)
        self.stack.push(self.stack.peek[instr.arg])

    def DUP_TOP(self, instr: Instruction):
        self.stack.push(self.stack.top)

    def DUP_TOP_TWO(self, instr: Instruction):
        for ref in self.stack.peek[:2]:
            self.stack.push(ref)

    def ROT_N(self, instr: Instruction):
        assert instr.argval is not None
        self._rot_top_n(instr.argval)

    def _rot_top_n(self, n: int):
        # a1 a2 a3 ... an  <- TOS
        # the stack changes to
        # an a1 a2 a3 an-1 <- TOS
        assert (
            len(self.stack) >= n
        ), f"There are not enough elements on the stack. {n} is needed."
        top = self.stack.pop()
        self.stack.insert(n - 1, top)

    def POP_TOP(self, instr: Instruction):
        self.stack.pop()

    def PUSH_NULL(self, instr: Instruction):
        self.stack.push(NullVariable())

    def ROT_TWO(self, instr: Instruction):
        self._rot_top_n(2)

    def ROT_THREE(self, instr: Instruction):
        self._rot_top_n(3)

    def ROT_FOUR(self, instr: Instruction):
        self._rot_top_n(4)

    def RESUME(self, instr: Instruction):
        # RESUME is a no-op, it just for internal tracing, debugging and optimization checks.
        pass

    def SWAP(self, instr: Instruction):
        assert isinstance(instr.arg, int)
        self.stack.top, self.stack.peek[instr.arg] = (
            self.stack.peek[instr.arg],
            self.stack.top,
        )

    # unary operators
    UNARY_POSITIVE = tos_op_wrapper(operator.pos)
    UNARY_NEGATIVE = tos_op_wrapper(operator.neg)
    UNARY_NOT = tos_op_wrapper(operator.not_)
    UNARY_INVERT = tos_op_wrapper(operator.invert)

    # binary operators
    BINARY_POWER = tos_op_wrapper(operator.pow)
    BINARY_MULTIPLY = tos_op_wrapper(operator.mul)
    BINARY_MATRIX_MULTIPLY = tos_op_wrapper(operator.matmul)
    BINARY_FLOOR_DIVIDE = tos_op_wrapper(operator.floordiv)
    BINARY_TRUE_DIVIDE = tos_op_wrapper(operator.truediv)
    BINARY_MODULO = tos_op_wrapper(operator.mod)
    BINARY_ADD = tos_op_wrapper(operator.add)
    BINARY_SUBTRACT = tos_op_wrapper(operator.sub)
    BINARY_LSHIFT = tos_op_wrapper(operator.lshift)
    BINARY_RSHIFT = tos_op_wrapper(operator.rshift)
    BINARY_AND = tos_op_wrapper(operator.and_)
    BINARY_OR = tos_op_wrapper(operator.or_)
    BINARY_XOR = tos_op_wrapper(operator.xor)

    def BINARY_OP(self, instr: Instruction):
        opname, _ = opcode._nb_ops[instr.arg]
        opname = (
            opname.replace("NB_", "BINARY_")
            .replace("BINARY_INPLACE", "INPLACE")
            .replace("REMAINDER", "MODULO")
        )
        return getattr(self, opname)(instr)

    @call_break_graph_decorator(push_n=1)
    def BINARY_SUBSCR(self, instr: Instruction):
        key = self.stack.pop()
        container = self.stack.pop()
        assert isinstance(key, VariableBase)
        # TODO(xiongkun): getitem / getattr support key and attr as variable.
        if isinstance(key, TensorVariable) and isinstance(
            container, TensorVariable
        ):
            # NOTE(xiongkun): tensor[tensor] should support.
            output = self._graph.call_tensor_method(
                "__getitem__", container, key
            )
            self.stack.push(output)
            return

        if isinstance(key, TensorVariable):
            raise BreakGraphError(
                f"Key is a TensorVariable in BINARY_SUBSCR, {container}[{key}]"
            )

        result = BuiltinVariable(
            operator.getitem, self._graph, DanglingTracker()
        )(container, key)
        self.stack.push(result)

    # inplace operators
    # paddle variable do not have inplace operators. For example when call `y **= x`, will call var.__pow__
    INPLACE_POWER = tos_inplace_op_wrapper(operator.ipow)
    INPLACE_MULTIPLY = tos_inplace_op_wrapper(operator.imul)
    INPLACE_MATRIX_MULTIPLY = tos_inplace_op_wrapper(operator.imatmul)
    INPLACE_FLOOR_DIVIDE = tos_inplace_op_wrapper(operator.ifloordiv)
    INPLACE_TRUE_DIVIDE = tos_inplace_op_wrapper(operator.itruediv)
    INPLACE_MODULO = tos_inplace_op_wrapper(operator.imod)
    INPLACE_ADD = tos_inplace_op_wrapper(operator.iadd)
    INPLACE_SUBTRACT = tos_inplace_op_wrapper(operator.isub)
    INPLACE_LSHIFT = tos_inplace_op_wrapper(operator.ilshift)
    INPLACE_RSHIFT = tos_inplace_op_wrapper(operator.irshift)
    INPLACE_AND = tos_inplace_op_wrapper(operator.iand)
    INPLACE_OR = tos_inplace_op_wrapper(operator.ior)
    INPLACE_XOR = tos_inplace_op_wrapper(operator.ixor)

    def NOP(self, instr: Instruction):
        pass

    @call_break_graph_decorator(push_n=1)
    def LOAD_ATTR(self, instr: Instruction):
        attr_name = self._code.co_names[instr.arg]
        attr_name_var = ConstantVariable.wrap_literal(attr_name, self._graph)
        obj = self.stack.pop()
        self.stack.push(
            BuiltinVariable(
                getattr, graph=self._graph, tracker=DanglingTracker()
            )(obj, attr_name_var)
        )

    def LOAD_CONST(self, instr: Instruction):
        var = self._co_consts[instr.arg]
        self.stack.push(var)

    def MAKE_CELL(self, instr: Instruction):
        self._locals[instr.argval] = self._cells[instr.argval]

    def LOAD_CLOSURE(self, instr: Instruction):
        if sys.version_info >= (3, 11):
            self.LOAD_FAST(instr)
            return
        namemap = self._code.co_cellvars + self._code.co_freevars
        name = namemap[instr.arg]
        self.stack.push(self._cells[name])

    def LOAD_DEREF(self, instr: Instruction):
        if sys.version_info >= (3, 11):
            self.stack.push(self._locals[instr.argval].cell_content())
            return
        namemap = self._code.co_cellvars + self._code.co_freevars
        name = namemap[instr.arg]
        self.stack.push(self._cells[name].cell_content())

    def COPY_FREE_VARS(self, instr: Instruction):
        for i in range(instr.arg):
            freevar_name = self._code.co_freevars[i]
            self._locals[freevar_name] = self._cells[freevar_name]

    def LOAD_FAST(self, instr: Instruction):
        var = self._locals[instr.argval]
        self.stack.push(var)

    def DELETE_FAST(self, instr: Instruction):
        varname = self._code.co_varnames[instr.arg]
        del self._locals[varname]

    def LOAD_GLOBAL(self, instr: Instruction):
        namei: int = instr.arg
        push_null = False
        if sys.version_info >= (3, 11):
            push_null = namei & 1
            namei >>= 1
        if push_null:
            self.stack.push(NullVariable())
        name = self._code.co_names[namei]
        if name in self._globals.keys():
            value = self._globals.get(name)
        elif name in self._builtins.keys():
            value = self._builtins[name]
        else:
            raise InnerError(f"{name} not in globals and builtins")
        self.stack.push(value)

    def LOAD_METHOD(self, instr: Instruction):
        method_name = self._code.co_names[instr.arg]
        method_name_var = ConstantVariable.wrap_literal(
            method_name, self._graph
        )
        obj = self.stack.pop()

        method = BuiltinVariable(
            getattr, graph=self._graph, tracker=DanglingTracker()
        )(obj, method_name_var)

        if isinstance(method, MethodVariable):
            # bound method, push the unbound method and the self
            self.stack.push(method.fn)
            self.stack.push(obj)
        else:
            # unbound method, push the dummy and the function
            self.stack.push(NullVariable())
            self.stack.push(method)

    @call_break_graph_decorator(push_n=0)
    def STORE_ATTR(self, instr: Instruction):
        obj = self.stack.pop()
        val = self.stack.pop()
        key = self._code.co_names[instr.arg]
        key_var = ConstantVariable.wrap_literal(key, self._graph)
        BuiltinVariable(
            setattr, self._graph, DummyTracker([obj, key_var, val])
        )(obj, key_var, val)

    def DELETE_ATTR(self, instr: Instruction):
        obj = self.stack.pop()
        key = instr.argval
        key_var = ConstantVariable.wrap_literal(key, self._graph)
        BuiltinVariable(delattr, self._graph, DummyTracker([obj, key_var]))(
            obj, key_var
        )

    def STORE_DEREF(self, instr: Instruction):
        if sys.version_info >= (3, 11):
            self._cells[instr.argval].set_value(self.stack.pop())
            self._locals[instr.argval] = self._cells[instr.argval]
            return
        namemap = self._code.co_cellvars + self._code.co_freevars
        name = namemap[instr.arg]
        self._cells[name].set_value(self.stack.pop())

    def STORE_FAST(self, instr: Instruction):
        """
        TODO: side effect may happen
        """
        var = self.stack.pop()
        name = self._code.co_varnames[instr.arg]
        var.debug_name = name
        self._locals[name] = var

    def STORE_GLOBAL(self, instr: Instruction):
        var = self.stack.pop()
        name = self._code.co_names[instr.arg]
        var.debug_name = name
        self._globals.set(name, var)

    def DELETE_GLOBAL(self, instr: Instruction):
        self._globals.delete(self._code.co_names[instr.arg])

    @call_break_graph_decorator(push_n=0)
    def STORE_SUBSCR(self, instr: Instruction):
        key = self.stack.pop()
        container = self.stack.pop()
        value = self.stack.pop()
        assert isinstance(key, VariableBase)
        self._graph.add_global_guarded_variable(key)
        if isinstance(key, TensorVariable):
            raise BreakGraphError(
                f"Key is a TensorVariable in STORE_SUBSCR, {container}[{key}] = {value}"
            )
        # TODO(xiongkun): support tensor[tensor] = tensor, dy2static is not the same with dygraph.
        container[key.get_py_value()] = value
        value.debug_name = f"{container.debug_name}[{key.debug_name}]"

    def DELETE_SUBSCR(self, instr: Instruction):
        key = self.stack.pop()
        container = self.stack.pop()
        assert isinstance(key, VariableBase)
        self._graph.add_global_guarded_variable(key)
        BuiltinVariable(operator.delitem, self._graph, DanglingTracker())(
            container, key
        )

    def BUILD_LIST(self, instr: Instruction):
        list_size = instr.arg
        assert list_size <= len(
            self.stack
        ), f"OpExecutor want BUILD_LIST with size {list_size}, but current stack do not have enough elems."
        val_list = self.stack.pop_n(list_size)
        self.stack.push(
            ListVariable(
                val_list, graph=self._graph, tracker=DummyTracker(val_list)
            )
        )

    def BUILD_TUPLE(self, instr: Instruction):
        tuple_size = instr.arg
        assert tuple_size <= len(
            self.stack
        ), f"OpExecutor want BUILD_TUPLE with size {tuple_size}, but current stack do not have enough elems."
        val_tuple = self.stack.pop_n(tuple_size)
        self.stack.push(
            TupleVariable(
                tuple(val_tuple),
                graph=self._graph,
                tracker=DummyTracker(val_tuple),
            )
        )

    def BUILD_STRING(self, instr: Instruction):
        count = instr.arg
        assert count <= len(
            self.stack
        ), f"OpExecutor want BUILD_STRING with size {count}, but current stack do not have enough elems."
        str_list = self.stack.pop_n(count)
        new_str = ''
        for s in str_list:
            assert s.get_py_type() is str
            new_str += s.get_py_value()
        self.stack.push(
            ConstantVariable(new_str, self._graph, DummyTracker(str_list))
        )

    @call_break_graph_decorator(push_n=1)
    def BUILD_SLICE(self, instr: Instruction):
        if instr.arg == 3:
            step = self.stack.pop()
        else:
            step = ConstantVariable.wrap_literal(None, self._graph)
        stop = self.stack.pop()
        start = self.stack.pop()

        self.stack.push(
            SliceVariable(
                slice(start, stop, step),
                graph=self._graph,
                tracker=DummyTracker([start, stop, step]),
            )
        )

    def build_map(
        self, keys: list[VariableBase], values: list[VariableBase]
    ) -> VariableBase:
        built_map = {}
        for key, value in zip(keys, values):
            assert isinstance(key, VariableBase)
            # Add key to global guarded variable to avoid missing the key guard
            self._graph.add_global_guarded_variable(key)
            key = key.get_py_value()
            built_map[key] = value
        return DictVariable(
            built_map,
            graph=self._graph,
            tracker=DummyTracker(keys + values),
        )

    def BUILD_MAP(self, instr: Instruction):
        map_size = instr.arg
        assert map_size * 2 <= len(
            self.stack
        ), f"OpExecutor want BUILD_MAP with size {map_size} * 2, but current stack do not have enough elems."
        val_for_dict = self.stack.pop_n(map_size * 2)
        keys = val_for_dict[::2]
        values = val_for_dict[1::2]
        self.stack.push(self.build_map(keys, values))

    def BUILD_CONST_KEY_MAP(self, instr: Instruction):
        map_size = instr.arg
        assert map_size + 1 <= len(
            self.stack
        ), f"OpExecutor want BUILD_CONST_KEY_MAP with size {map_size} + 1, but current stack do not have enough elems."
        keys = self.stack.pop().get_items()
        assert len(keys) == map_size
        values = self.stack.pop_n(map_size)
        self.stack.push(self.build_map(keys, values))

    def build_seq_unpack(self, instr: Instruction):
        oparg = instr.arg
        assert isinstance(oparg, int)
        unpack_values = self.stack.pop_n(oparg)

        retval = []
        for item in unpack_values:
            assert isinstance(item, (TupleVariable, ListVariable))
            retval.extend(item.get_wrapped_items())

        if instr.opname in {
            "BUILD_TUPLE_UNPACK_WITH_CALL",
            "BUILD_TUPLE_UNPACK",
        }:
            retval = tuple(retval)

        self.stack.push(
            VariableFactory.from_value(
                retval, self._graph, DummyTracker(unpack_values)
            )
        )

    def BUILD_TUPLE_UNPACK_WITH_CALL(self, instr: Instruction):
        self.build_seq_unpack(instr)

    def BUILD_TUPLE_UNPACK(self, instr: Instruction):
        self.build_seq_unpack(instr)

    def BUILD_LIST_UNPACK(self, instr: Instruction):
        self.build_seq_unpack(instr)

    def BUILD_MAP_UNPACK(self, instr: Instruction):
        oparg = instr.arg
        assert isinstance(oparg, int)
        unpack_values = self.stack.pop_n(oparg)

        retval = {}
        for item in unpack_values:
            assert item.get_py_type() is dict
            retval.update(item.get_wrapped_items())

        self.stack.push(
            VariableFactory.from_value(
                retval, self._graph, DummyTracker(unpack_values)
            )
        )

    def BUILD_MAP_UNPACK_WITH_CALL(self, instr: Instruction):
        oparg = instr.arg
        assert isinstance(oparg, int)
        unpack_values = self.stack.pop_n(oparg)

        retval = {}
        for item in unpack_values:
            assert item.get_py_type() is dict
            wrapped_item = item.get_wrapped_items()
            if wrapped_item.items() & retval.items():
                raise InnerError(
                    "BUILD_MAP_UNPACK_WITH_CALL found repeated key."
                )
            retval.update(wrapped_item)

        self.stack.push(
            VariableFactory.from_value(
                retval, self._graph, DummyTracker(unpack_values)
            )
        )

    def PRECALL(self, instr: Instruction):
        assert isinstance(instr.arg, int)
        is_method_layout = not isinstance(
            self.stack.peek[instr.arg + 2], NullVariable
        )
        nargs = instr.arg + int(is_method_layout)
        method = self.stack.peek[nargs + 1]
        if not is_method_layout and isinstance(method, MethodVariable):
            unbound_method = method.fn
            self_var = method.bound_instance
            self.stack.peek[nargs + 1] = self_var
            self.stack.peek[nargs + 2] = unbound_method

    def KW_NAMES(self, instr: Instruction):
        assert self._call_shape is None
        assert isinstance(instr.arg, int)
        self._call_shape = self._co_consts[instr.arg].get_py_value()

    @call_break_graph_decorator(push_n=1)
    def CALL(self, instr: Instruction):
        assert isinstance(instr.arg, int)
        assert instr.arg + 2 <= len(self.stack)
        is_method = not isinstance(self.stack.peek[instr.arg + 2], NullVariable)
        total_args = instr.arg + int(is_method)
        kwnames = self._call_shape if self._call_shape is not None else []
        n_kwargs = len(kwnames)
        n_positional_args = total_args - n_kwargs
        kwargs_list = self.stack.pop_n(n_kwargs)
        kwargs = dict(zip(kwnames, kwargs_list))
        args = self.stack.pop_n(n_positional_args)
        fn = self.stack.pop()
        if not is_method:
            # pop the NULL variable
            self.stack.pop()
        self.stack.push(fn(*args, **kwargs))
        self._call_shape = None

    @call_break_graph_decorator(push_n=1)
    def CALL_FUNCTION(self, instr: Instruction):
        assert isinstance(instr.arg, int)
        n_args = instr.arg
        assert isinstance(n_args, int)
        args = self.stack.pop_n(n_args)
        kwargs = {}
        fn = self.stack.pop()
        ret = fn(*args, **kwargs)
        self.stack.push(ret)

    @call_break_graph_decorator(push_n=1)
    def CALL_FUNCTION_KW(self, instr: Instruction):
        n_args = instr.arg
        assert n_args + 2 <= len(self.stack)

        kwargs_keys = self.stack.pop()
        assert isinstance(kwargs_keys, TupleVariable)
        assert len(kwargs_keys) > 0
        kwargs_keys = [
            x.get_py_value() if isinstance(x, VariableBase) else x
            for x in kwargs_keys.get_py_value()
        ]

        # split arg_list to args and kwargs
        arg_list = self.stack.pop_n(n_args)
        args = arg_list[: -len(kwargs_keys)]
        kwargs_values = arg_list[-len(kwargs_keys) :]
        kwargs = dict(zip(kwargs_keys, kwargs_values))

        fn = self.stack.pop()
        ret = fn(*args, **kwargs)
        self.stack.push(ret)

    @call_break_graph_decorator(push_n=1)
    def CALL_FUNCTION_EX(self, instr: Instruction):
        flag = instr.arg
        if flag & CFE.CFE_HAS_KWARGS:
            kwargs_variable = self.stack.pop()
            assert isinstance(kwargs_variable, DictVariable)
            kwargs = kwargs_variable.get_wrapped_items()
        else:
            kwargs = {}

        args_variable = self.stack.pop()
        assert isinstance(args_variable, (TupleVariable, ListVariable))
        args = args_variable.get_wrapped_items()

        fn = self.stack.pop()
        if sys.version_info >= (3, 11):
            null = self.stack.pop()
            assert isinstance(null, NullVariable)
        ret = fn(*args, **kwargs)
        self.stack.push(ret)

    @call_break_graph_decorator(push_n=1)
    def CALL_METHOD(self, instr: Instruction):
        n_args = instr.arg
        assert isinstance(n_args, int)
        args = self.stack.pop_n(n_args)
        self_var = self.stack.pop()
        method = self.stack.pop()
        if isinstance(method, NullVariable):
            method = self_var
        else:
            args = [self_var] + args
        self.stack.push(method(*args))

    @call_break_graph_decorator(
        push_n=1
    )  # call instance, in, not in may call TensorVariable.get_py_value, which raise BreakGraphError
    def COMPARE_OP(self, instr: Instruction):
        op = dis.cmp_op[instr.arg]
        right, left = self.stack.pop(), self.stack.pop()
        self.stack.push(
            BuiltinVariable(
                SUPPORT_COMPARE_OP[op], self._graph, DanglingTracker()
            )(left, right)
        )

    @call_break_graph_decorator(push_n=1)
    def IS_OP(self, instr: Instruction):
        # It will only be 0 or 1
        assert instr.arg == 0 or instr.arg == 1
        right, left = self.stack.pop(), self.stack.pop()
        op = "is" if instr.arg == 0 else "is not"
        self.stack.push(
            BuiltinVariable(
                SUPPORT_COMPARE_OP[op], self._graph, DanglingTracker()
            )(left, right)
        )

    def MAKE_FUNCTION(self, instr: Instruction):
        if sys.version_info < (3, 11):
            fn_name = self.stack.pop()
        codeobj = self.stack.pop()
        if sys.version_info >= (3, 11):
            # MAKE_FUNCTION behavior actually changed in 3.11, see
            # https://github.com/python/cpython/pull/93189/
            assert hasattr(codeobj.value, "co_qualname")
            fn_name = ConstantVariable(
                codeobj.value.co_qualname, self._graph, DummyTracker([codeobj])
            )

        global_dict = self._globals.get_value()

        related_list = [fn_name, codeobj]

        flag = instr.arg
        if flag & MF.MF_HAS_CLOSURE:
            # closure should be a tuple of Variables
            closure_variable = self.stack.pop()
            assert isinstance(closure_variable, TupleVariable)
            closure = []
            for item in closure_variable.get_wrapped_items():
                closure.append(types.CellType())
                closure[-1].cell_contents = item
            closure = tuple(closure)
        else:
            closure = ()

        if flag & MF.MF_HAS_ANNOTATION:
            # can not set annotation in python env, skip it
            related_list.append(self.stack.pop())

        if flag & MF.MF_HAS_KWDEFAULTS:
            raise FallbackError(
                "Found need func_kwdefaults when MAKE_FUNCTION."
            )

        if flag & MF.MF_HAS_DEFAULTS:
            '''
            default_args should have tracker too, like:

            def f(x):
                def g(z=x):
                    pass
            '''
            default_args_variable = self.stack.pop()
            assert isinstance(default_args_variable, TupleVariable)
            related_list.append(default_args_variable)
            default_args = tuple(default_args_variable.get_wrapped_items())
        else:
            default_args = ()

        new_fn = types.FunctionType(
            codeobj.get_py_value(),
            global_dict,
            fn_name.get_py_value(),
            default_args,
            closure,
        )
        self.stack.push(
            UserDefinedFunctionVariable(
                new_fn, self._graph, DummyTracker(related_list)
            )
        )

    def GET_ITER(self, instr: Instruction):
        source_obj = self.stack.pop()
        iter_variable = BuiltinVariable(iter, self._graph, DanglingTracker())(
            source_obj
        )
        self.stack.push(iter_variable)

    def JUMP_ABSOLUTE(self, instr: Instruction):
        assert instr.jump_to is not None
        self.jump_to(instr.jump_to)

    def JUMP_FORWARD(self, instr: Instruction):
        self.JUMP_ABSOLUTE(instr)

    def JUMP_BACKWARD(self, instr: Instruction):
        # TODO: check interrupt
        self.JUMP_ABSOLUTE(instr)

    def JUMP_BACKWARD_NO_INTERRUPT(self, instr: Instruction):
        self.JUMP_ABSOLUTE(instr)

    @call_break_graph_decorator(push_n=1)
    def CONTAINS_OP(self, instr: Instruction):
        # It will only be 0 or 1
        assert instr.arg == 0 or instr.arg == 1
        right, left = self.stack.pop(), self.stack.pop()
        op = "in" if instr.arg == 0 else "not in"
        self.stack.push(
            BuiltinVariable(
                SUPPORT_COMPARE_OP[op], self._graph, DanglingTracker()
            )(left, right)
        )

    @jump_break_graph_decorator
    def JUMP_IF_FALSE_OR_POP(self, instr: Instruction):
        pred_obj = self.stack.top
        if isinstance(pred_obj, (ConstantVariable, ContainerVariable)):
            self._graph.add_global_guarded_variable(pred_obj)
            is_jump = not bool(pred_obj)
            if is_jump:
                assert instr.jump_to is not None
                self.jump_to(instr.jump_to)
            else:
                self.stack.pop()
            return
        raise FallbackError(
            "Currently don't support predicate a non-const / non-tensor obj."
        )

    @jump_break_graph_decorator
    def JUMP_IF_TRUE_OR_POP(self, instr: Instruction):
        pred_obj = self.stack.top
        if isinstance(pred_obj, (ConstantVariable, ContainerVariable)):
            self._graph.add_global_guarded_variable(pred_obj)
            is_jump = bool(pred_obj)
            if is_jump:
                assert instr.jump_to is not None
                self.jump_to(instr.jump_to)
            else:
                self.stack.pop()
            return
        raise FallbackError(
            "Currently don't support predicate a non-const / non-tensor obj."
        )

    POP_JUMP_IF_FALSE = pop_jump_if_op_wrapper([bool, operator.not_])
    POP_JUMP_FORWARD_IF_FALSE = POP_JUMP_IF_FALSE
    POP_JUMP_BACKWARD_IF_FALSE = POP_JUMP_IF_FALSE

    POP_JUMP_IF_TRUE = pop_jump_if_op_wrapper([bool])
    POP_JUMP_FORWARD_IF_TRUE = POP_JUMP_IF_TRUE
    POP_JUMP_BACKWARD_IF_TRUE = POP_JUMP_IF_TRUE

    POP_JUMP_FORWARD_IF_NONE = pop_jump_if_op_wrapper([operator_is_none])
    POP_JUMP_BACKWARD_IF_NONE = POP_JUMP_FORWARD_IF_NONE

    POP_JUMP_FORWARD_IF_NOT_NONE = pop_jump_if_op_wrapper(
        [operator_is_not_none]
    )
    POP_JUMP_BACKWARD_IF_NOT_NONE = POP_JUMP_FORWARD_IF_NOT_NONE

    @call_break_graph_decorator(push_n=lambda arg: arg)
    def UNPACK_SEQUENCE(self, instr: Instruction):
        sequence = self.stack.pop()
        seq_iter = BuiltinVariable(iter, self._graph, DanglingTracker())(
            sequence
        )
        unpacked = []
        for _ in range(instr.arg):
            unpacked.append(seq_iter.next())
        for item in reversed(unpacked):
            self.stack.push(item)

    def UNPACK_EX(self, instr: Instruction):
        getitem = BuiltinVariable(
            operator.getitem, self._graph, DanglingTracker()
        )
        assert instr.arg is not None
        sequence = self.stack.pop()
        if not isinstance(
            sequence, (ListVariable, TupleVariable, TensorVariable)
        ):
            raise FallbackError(f"Unpack {sequence} is not implemented.")

        if instr.argval >= 256:
            # NOTE: If the number of unpacked variables exceeds 256, python will report an error like:
            # SyntaxError: too many expressions in star-unpacking assignmen,
            # so if the number of unpacked variables exceeds 256, it will be treated as the following case.
            # a, b, *c, d = e
            front_nums = instr.arg & 0xFF
            back_nums = instr.arg >> 8
            assert (
                len(sequence) >= front_nums + back_nums
            ), f"Want unpack {sequence} to {front_nums + back_nums}, but {len(sequence)} is smaller than {front_nums + back_nums}."

            for i in range(
                len(sequence) - 1, len(sequence) - back_nums - 1, -1
            ):
                self.stack.push(getitem(sequence, i))

            slice_var = SliceVariable(
                slice(front_nums, len(sequence) - back_nums - 1),
                self._graph,
                DummyTracker([sequence]),
            )
        else:
            # a, b, c, *d = e
            assert (
                len(sequence) >= instr.arg
            ), f"Want unpack {sequence} to {instr.arg}, but {len(sequence)} is smaller than {instr.arg}."

            slice_obj = slice(instr.arg, None)
            slice_var = SliceVariable(
                slice_obj, self._graph, ConstTracker(slice_obj)
            )
            front_nums = instr.arg
        self.stack.push(getitem(sequence, slice_var))
        for i in range(front_nums - 1, -1, -1):
            self.stack.push(getitem(sequence, i))

    def FORMAT_VALUE(self, instr: Instruction):
        flag = instr.arg
        assert flag is not None
        which_conversion = flag & FV.FVC_MASK
        have_fmt_spec = bool((flag & FV.FVS_MASK) == FV.FVS_HAVE_SPEC)

        fmt_spec = self.stack.pop().get_py_value() if have_fmt_spec else ""
        value = self.stack.pop()

        if which_conversion == FV.FVC_NONE:
            convert_fn = None
        elif which_conversion == FV.FVC_STR:
            convert_fn = "__str__"
        elif which_conversion == FV.FVC_REPR:
            convert_fn = "__repr__"
        elif which_conversion == FV.FVC_ASCII:
            convert_fn = "__ascii__"
        else:
            raise InnerError(
                f"Unexpected conversion flag {flag} for FORMAT_VALUE"
            )

        # different type will lead to different Tracker, so call self.stack.push in different branch
        if isinstance(value, ConstantVariable):
            result = value.get_py_value()
            if convert_fn is not None:
                result = getattr(result, convert_fn)(result)

            if not isinstance(result, str) or fmt_spec != "":
                result = format(result, fmt_spec)

            self.stack.push(
                ConstantVariable(result, self._graph, DummyTracker([value]))
            )
        else:
            raise FallbackError(f"Do not support format {type(value)} now")

    # NOTE: This operation will generate SideEffects, and the mechanism has not been completed yet
    def DICT_UPDATE(self, instr: Instruction):
        dict_value = self.stack.pop()
        assert isinstance(instr.arg, int)
        BuiltinVariable(dict.update, self._graph, tracker=DanglingTracker())(
            self.stack.peek[instr.arg], dict_value
        )

    def DICT_MERGE(self, instr: Instruction):
        dict_value = self.stack.pop()
        assert isinstance(instr.arg, int)
        for key in dict_value.get_wrapped_items().keys():
            result = (
                self.stack.peek[instr.arg].get_wrapped_items().get(key, None)
            )
            if result is not None:
                raise InnerError(
                    f"got multiple values for keyword argument '{key}'"
                )
        BuiltinVariable(dict.update, self._graph, tracker=DanglingTracker())(
            self.stack.peek[instr.arg], dict_value
        )

    def LIST_APPEND(self, instr: Instruction):
        list_value = self.stack.pop()
        assert isinstance(instr.arg, int)
        BuiltinVariable(list.append, self._graph, tracker=DanglingTracker())(
            self.stack.peek[instr.arg], list_value
        )

    def MAP_ADD(self, instr: Instruction):
        key, value = self.stack.pop_n(2)
        assert isinstance(instr.arg, int)
        BuiltinVariable(operator.setitem, self._graph, DanglingTracker())(
            self.stack.peek[instr.arg], key, value
        )

    def LIST_EXTEND(self, instr: Instruction):
        list_value = self.stack.pop()
        assert isinstance(instr.arg, int)
        BuiltinVariable(list.extend, self._graph, tracker=DanglingTracker())(
            self.stack.peek[instr.arg], list_value
        )

    def LIST_TO_TUPLE(self, instr: Instruction):
        list_value = self.stack.pop()
        self.stack.push(
            TupleVariable(
                list_value.get_wrapped_items(),
                self._graph,
                DummyTracker([list_value]),
            )
        )


class OpcodeExecutor(OpcodeExecutorBase):
    """
    A class that represents an executor for opcode operations.

    Args:
        frame: The frame object.

    """

    def __init__(self, frame: types.FrameType, **kwargs):
        graph = FunctionGraph(frame, **kwargs)
        self._frame = frame
        self._name = "Executor"
        self.call_stack[:] = []
        super().__init__(frame.f_code, graph)
        Dispatcher.graph = graph

    def cleanup(self):
        self._graph.pycode_gen = None
        Dispatcher.graph = None

    @event_register("OpcodeExecutor: _prepare_virtual_env", event_level=2)
    def _prepare_virtual_env(self):
        """
        Prepare the virtual environment for execution by adding variables from locals, globals, builtins, and constants.

        """
        log(
            3,
            f"[Executor] code options: co_cellvars={self._frame.f_code.co_cellvars}\n",
        )
        free_or_cell_vars = (
            self._frame.f_code.co_cellvars + self._frame.f_code.co_freevars
        )
        for name, value in self._frame.f_locals.items():
            tracker = (
                CellTracker(name)
                if name in free_or_cell_vars
                else LocalTracker(name)
            )
            self._locals[name] = VariableFactory.from_value(
                value, self._graph, tracker, debug_name=name
            )

        for name in free_or_cell_vars:
            # create a cell for each variable.
            self._cells[name] = CellVariable()  # put in cells.
            if name in self._locals:
                self._cells[name].set_value(self._locals[name])

        self._globals = GlobalVariable(
            self._frame.f_globals,
            self._graph,
            DanglingTracker(),
        )

        self._builtins = self._graph._builtins

        for value in self._code.co_consts:
            self._co_consts.append(
                VariableFactory.from_value(
                    value, self._graph, ConstTracker(value)
                )
            )

    def _create_resume_fn(self, index, stack_size=0):
        """
        Create a resume function and its inputs at the specified index.

        Args:
            index: The index at which the resume function is created.
            stack_size: The size of the stack.

        Returns:
            The resume function and its inputs.

        """
        pycode_gen = PyCodeGen(self._frame)
        fn, inputs = pycode_gen.gen_resume_fn_at(index, stack_size)
        return fn, inputs

    @fallback_when_occur_error
    def _break_graph_in_jump(self, result: VariableBase, instr: Instruction):
        """
        Break the graph at a JUMP instruction.

        Args:
            result: The result variable of the jump instruction.
            instr: The jump instruction.

        """
        self._graph.add_global_guarded_variable(result)
        stack_size = len(self.stack)
        if_fn, if_inputs = self._create_resume_fn(
            self.indexof(instr) + 1, stack_size
        )
        else_fn, else_inputs = self._create_resume_fn(
            self.indexof(instr.jump_to), stack_size
        )

        # gen call static fn opcode
        inputs_name = if_inputs | else_inputs
        inputs_var = [
            self.get_var(name)
            for name in inputs_name
            if self.get_var(name) is not result
        ]
        ret_vars = [
            result,
        ] + inputs_var
        # Collect all the to store variables.
        store_vars = []
        for stack_arg in self.stack:
            store_vars.append(stack_arg)
        for name in inputs_name:
            store_vars.append(self.get_var(name))

        var_loader = self._graph.start_compile_with_name_store(
            ret_vars, store_vars
        )
        # only pop the input of if/else resume fn, and keep the bool tensor result on the stack
        for _ in inputs_var:
            self._graph.pycode_gen.gen_pop_top()

        # gen call if/else resume fn opcode
        if if_fn is not None:
            self._graph.pycode_gen.gen_load_object(
                if_fn, if_fn.__code__.co_name
            )
            insert_index = len(self._graph.pycode_gen._instructions) - 1
            for i, stack_arg in enumerate(self.stack):
                var_loader.load(
                    stack_arg, allow_push_null=i >= len(self.stack) - 1
                )
            for name in if_inputs:
                var_loader.load(self.get_var(name))
            self._graph.pycode_gen.gen_call_function(
                argc=if_fn.__code__.co_argcount,
            )
            self._graph.pycode_gen.gen_return()
        else:
            insert_index = len(self._graph.pycode_gen._instructions) - 1
            self._graph.pycode_gen.gen_return()

        if else_fn is not None:
            self._graph.pycode_gen.gen_load_object(
                else_fn, else_fn.__code__.co_name
            )
            jump_to = self._graph.pycode_gen._instructions[-1]
            for i, stack_arg in enumerate(self.stack):
                var_loader.load(
                    stack_arg, allow_push_null=i >= len(self.stack) - 1
                )
            for name in else_inputs:
                var_loader.load(self.get_var(name))
            self._graph.pycode_gen.gen_call_function(
                argc=else_fn.__code__.co_argcount,
            )
            self._graph.pycode_gen.gen_return()
        else:
            self._graph.pycode_gen.gen_return()
            jump_to = self._graph.pycode_gen._instructions[-1]

        # gen jump opcode
        self._graph.pycode_gen._insert_instr(
            insert_index, instr.opname, jump_to=jump_to
        )

        self.new_code = self._graph.pycode_gen.gen_pycode()
        self.guard_fn = self._graph.guard_fn

    @fallback_when_occur_error
    def _break_graph_in_call(
        self,
        origin_stack: VariableStack,
        instr: Instruction,
        push_n: int | Callable[[int | None], int],
    ):
        """
        Break the graph at a CALL instruction.

        Args:
            origin_stack: The original stack.
            instr: The call instruction.
            push_n: The number of elements to be pushed onto the stack.

        """
        push_n = push_n(instr.arg) if callable(push_n) else push_n
        index = self.indexof(instr)
        self.stack = origin_stack

        # gen call static fn opcode
        ret_vars = [
            arg
            for arg in self.stack
            if isinstance(arg, (TensorVariable, ContainerVariable))
        ]
        resume_input_name = analysis_inputs(self._instructions, index + 1)
        ret_vars = ret_vars + [
            self.get_var(name)
            for name in resume_input_name
            if self.get_var(name) not in ret_vars
        ]

        # Collect all the to store variables.
        store_vars = []
        for stack_arg in self.stack:
            store_vars.append(stack_arg)
        for name in resume_input_name:
            store_vars.append(self.get_var(name))
        var_loader = self._graph.start_compile_with_name_store(
            ret_vars, store_vars
        )

        for _ in ret_vars:
            self._graph.pycode_gen.gen_pop_top()

        # gen graph break call fn opcode
        stack_effect = calc_stack_effect(instr)
        pop_n = push_n - stack_effect

        for i, stack_arg in enumerate(self.stack):
            var_loader.load(
                stack_arg, allow_push_null=i >= len(self.stack) - pop_n
            )

        # gen call resume fn opcode
        # NOTE(SigureMo): In Python 3.11，we need generate KW_NAMES if the call shape is not None.
        self._graph.pycode_gen.gen_kw_names(self._call_shape)
        self._graph.pycode_gen.add_pure_instructions([instr])
        self.stack.pop_n(pop_n)
        stack_size = len(self.stack) + push_n

        resume_fn, _ = self._create_resume_fn(index + 1, stack_size)
        if resume_fn:
            self._graph.pycode_gen.gen_load_object(
                resume_fn, resume_fn.__code__.co_name
            )
            # NOTE(zrr1999): We need to shift the resume_fn under its arguments.
            # In Python 3.11+, NULL + resume_fn should be shifted together.
            shift_n = 2 if sys.version_info >= (3, 11) else 1
            self._graph.pycode_gen.gen_shift_n(shift_n, stack_size + shift_n)
            for name in resume_input_name:
                var_loader.load(self.get_var(name))
            self._graph.pycode_gen.gen_call_function(
                argc=resume_fn.__code__.co_argcount,
            )

        # gen RETURN_VALUE
        self._graph.pycode_gen.gen_return()

        self.new_code = self._graph.pycode_gen.gen_pycode()
        self.guard_fn = self._graph.guard_fn

    def transform(self):
        self.run()
        if self.new_code is None:
            raise InnerError("OpExecutor return a empty new_code.")
        # stopped by RETURN_VALUE and has sir len is enough => disable_eval_frame
        simulate_complete = bool(self.stop_state == "Return")
        if simulate_complete:
            if self._graph.sir_ctx.TOS.graph_size() < ENV_MIN_GRAPH_SIZE.get():
                raise FallbackError(
                    "Fallback after simulate for reasons.",
                    disable_eval_frame=True,
                )
            else:
                # if simulate stop with graph successfully, the all codes will be
                # surrounded by the eval_frame triggers which exist in self.new_code
                # we need not set disable_eval_frame=False here (for it already is)
                return (
                    CustomCode(self.new_code, True),
                    self.guard_fn,
                )
        else:
            # if return because breakgraph, need open eval_frame
            return (
                CustomCode(self.new_code, False),
                self.guard_fn,
            )

    def _gen_loop_body_between(
        self, inputs: list, for_iter_idx: int, start: int, end: int
    ) -> types.FunctionType:
        """
        Generates the loop body between the specified indices in the instruction list.

        Args:
            inputs: function inputs infos
            for_iter_idx (int): For find the for_iter opcode
            start (int): The start index of the loop body.
            end (int): The end index of the loop body.

        Returns:
            tuple: The generated loop body function object and its inputs.

        """
        pycode_gen = PyCodeGen(self._frame)
        origin_instrs = get_instructions(pycode_gen._origin_code)

        for_iter = origin_instrs[for_iter_idx]

        # for balance the stack (the loop body will pop iter first before break or return)
        # this None is used for replace the iterator obj in stack top
        pycode_gen.gen_load_const(None)

        # extend loop body main logic
        pycode_gen.extend_instrs(origin_instrs[start:end])

        # break should jump to this nop
        nop_for_break = pycode_gen._add_instr("NOP")

        # need do additional operates when break
        pycode_gen.gen_load_const(False)
        pycode_gen.gen_store_fast(inputs[-1])
        pycode_gen.gen_load_const(None)  # keep stack balance

        # continue should jump to this nop
        nop_for_continue = pycode_gen._add_instr("NOP")
        pycode_gen.gen_pop_top()

        # relocate jump
        out_loop = for_iter.jump_to
        for instr in pycode_gen._instructions:
            if instr.jump_to == for_iter:
                instr.jump_to = nop_for_continue
            if instr.jump_to == out_loop:
                instr.jump_to = nop_for_break

        # outputs is the same as inputs
        pycode_gen.gen_outputs_and_return(inputs)
        return pycode_gen.create_fn_with_inputs(inputs)

    @fallback_when_occur_error
    def _break_graph_in_for_loop(
        self, iterator: VariableBase, for_iter: Instruction
    ):
        '''
        for_iter: the FOR_ITER opcode

        need find out opcodes which unpack value from FOR_ITER, by analysing stack

        case 1:
            for i in iter:

            FOR_ITER
            STORE_FAST i

        case 2:
            for i,j in iter:

            FOR_ITER
            UNPACK_SEQUENCE 2
            STORE_FAST i
            STORE_FAST j

        TODO: check var is in globals or builtins, only locals considered now
        '''
        # 0. prepare sub functions
        # 0.1 find the range of loop body
        assert for_iter.jump_to is not None
        loop_body_start_idx = self.indexof(for_iter) + 1
        loop_body_end_idx = self.indexof(for_iter.jump_to)
        curent_stack = 1

        while True:
            if loop_body_start_idx >= len(self._instructions):
                raise InnerError("Can not balance stack in loop body.")
            cur_instr = self._instructions[loop_body_start_idx]
            # do not consider jump instr
            stack_effect = calc_stack_effect(cur_instr, jump=False)
            curent_stack += stack_effect
            loop_body_start_idx += 1
            if curent_stack == 0:
                break

        # 0.2 create loop body function
        all_used_vars = analysis_used_names_with_space(
            self._instructions, loop_body_start_idx, loop_body_end_idx
        )
        loop_body_inputs = [
            k
            for k, v in all_used_vars.items()
            if v in (Space.locals, Space.cells)
        ] + ["_break_flag"]

        loop_body_fn = self._gen_loop_body_between(
            loop_body_inputs,
            self.indexof(for_iter),
            loop_body_start_idx,
            loop_body_end_idx,
        )

        log(3, "[Resumed Function]: break graph in loop create loop body as\n")
        log_do(3, lambda: dis.dis(loop_body_fn))

        # 0.3 create after loop part function
        after_loop_fn, fn_inputs = self._create_resume_fn(
            loop_body_end_idx, len(self.stack)
        )

        total_inputs = OrderedSet(list(fn_inputs) + list(loop_body_inputs[:-1]))

        # 1. part before for-loop, start compile
        ret_names = [
            name
            for name in total_inputs
            if name in chain(self._locals, self._cells)
        ]
        ret_vars = [self.get_var(name) for name in ret_names]
        store_vars = [ret_vars[idx] for idx in range(len(ret_names))]
        store_vars.extend(iter(self.stack))
        store_vars.append(iterator.get_hold())
        var_loader = self._graph.start_compile_with_name_store(
            ret_vars, store_vars
        )

        for _ in ret_vars:
            self._graph.pycode_gen.gen_pop_top()

        # 2. restore vars
        for idx in range(len(ret_names)):
            var_loader.load(ret_vars[idx])
            self._graph.pycode_gen.gen_store(ret_names[idx], self._code)

        # 3. setup vars which is created in loop
        undefined_names = set()
        for name in loop_body_inputs[:-1]:
            if not self.has_var(name, all_used_vars[name]):
                undefined_names.add(name)
                self._graph.pycode_gen.gen_load_const(SotUndefinedVar())
                self._graph.pycode_gen.gen_store(name, self._code)

        # close eval_frame
        # TODO: need support effective strategies
        # self._graph.pycode_gen.gen_disable_eval_frame()

        # 4.1 load iterator
        iterator.reconstruct(self._graph.pycode_gen)

        # 4.2 gen FOR_ITER and unpack data
        self._graph.pycode_gen.extend_instrs(
            self._instructions[self.indexof(for_iter) : loop_body_start_idx]
        )

        # 5. call loop body
        # 5.1 load loop body
        self._graph.pycode_gen.gen_load_object(
            loop_body_fn, loop_body_fn.__code__.co_name
        )

        # 5.2 load loop body inputs
        for name in loop_body_inputs[:-1]:
            self._graph.pycode_gen.gen_load(name)

        # 5.3 load break flag
        self._graph.pycode_gen.gen_load_const(True)

        # 5.4 call loop body
        self._graph.pycode_gen.gen_call_function(
            argc=loop_body_fn.__code__.co_argcount
        )

        # 5.5 unpack and store retval, keep break_flag in stack
        self._graph.pycode_gen.gen_unpack_sequence(len(loop_body_inputs))

        for name in loop_body_inputs[:-1]:
            self._graph.pycode_gen.gen_store(name, self._code)

        # 6. add jump if break
        jump_if_break = self._graph.pycode_gen.gen_pop_jump(
            direction=JumpDirection.FORWARD, suffix=PopJumpCond.FALSE
        )

        # 7. jump back to FOR_ITER
        self._graph.pycode_gen.gen_jump(
            for_iter, direction=JumpDirection.BACKWARD
        )
        nop = self._graph.pycode_gen._add_instr("NOP")
        for_iter.jump_to = nop
        jump_if_break.jump_to = nop

        # open eval_frame
        # TODO: need support effective strategies
        # self._graph.pycode_gen.gen_enable_eval_frame()

        # 8. call after_loop_fn
        self._graph.pycode_gen.gen_load_object(
            after_loop_fn, after_loop_fn.__code__.co_name
        )

        for stack_arg in self.stack:
            var_loader.load(stack_arg)
        for name in fn_inputs:
            if not self.has_var(name) and name not in undefined_names:
                undefined_names.add(name)
                self._graph.pycode_gen.gen_load_const(SotUndefinedVar())
                self._graph.pycode_gen.gen_store(name, self._code)
            self._graph.pycode_gen.gen_load(name)

        self._graph.pycode_gen.gen_call_function(
            argc=after_loop_fn.__code__.co_argcount
        )

        self._graph.pycode_gen.gen_return()
        self.new_code = self._graph.pycode_gen.gen_pycode()
        self.guard_fn = self._graph.guard_fn

    def _inline_call_for_loop(
        self, iterator: VariableBase, for_iter: Instruction
    ):
        assert for_iter.jump_to is not None
        pycode_gen = PyCodeGen(self._frame)
        origin_instrs = get_instructions(pycode_gen._origin_code)

        start_idx = self.indexof(for_iter)
        end_idx = self.indexof(for_iter.jump_to)

        all_used_vars = analysis_used_names_with_space(
            origin_instrs, start_idx, end_idx
        )

        inputs = [
            k
            for k, v in all_used_vars.items()
            if v in (Space.locals, Space.cells)
        ] + [iterator.id]

        # 1. load iter
        pycode_gen.gen_load_fast(iterator.id)

        # 2. copy main logic
        pycode_gen.extend_instrs(origin_instrs[start_idx:end_idx])

        # 3. add break, continue marker and relocate jump
        for_iter_instr = origin_instrs[start_idx]
        assert for_iter_instr.jump_to is not None
        out_loop_instr = for_iter_instr.jump_to

        pycode_gen.gen_jump(out_loop_instr, direction=JumpDirection.FORWARD)
        nop_for_continue = pycode_gen._add_instr("NOP")

        jump = pycode_gen.gen_jump(
            for_iter_instr, direction=JumpDirection.BACKWARD
        )

        nop_for_break = pycode_gen._add_instr("NOP")

        for instr in pycode_gen._instructions:
            if instr.jump_to == for_iter_instr:
                instr.jump_to = nop_for_continue

            if (
                instr.jump_to in origin_instrs
                and origin_instrs.index(instr.jump_to) >= end_idx
            ):
                instr.jump_to = nop_for_break

        jump.jump_to = for_iter_instr
        pycode_gen.gen_outputs_and_return(inputs)
        inline_call_fn = pycode_gen.create_fn_with_inputs(inputs)

        log(
            3,
            f"[Resumed Function]: Inline call for loop function {inline_call_fn.__code__.co_name}\n",
        )
        log_do(3, lambda: dis.dis(inline_call_fn))

        # TODO: update globals builtins
        fn = UserDefinedFunctionVariable(
            inline_call_fn,
            self._graph,
            DanglingTracker(),
        )

        input_vars = [
            self.get_var(name)
            if self.has_var(name, all_used_vars[name])
            else SotUndefinedVar()
            for name in inputs[:-1]
        ] + [iterator]
        ret = fn(*input_vars)
        # slice_variable is [:-1]
        slice_const = slice(None, -1, None)
        slice_variable = SliceVariable(
            slice_const, self._graph, ConstTracker(slice_const)
        )
        for name, val in zip(inputs[:-1], ret[slice_variable]):
            self._locals[name] = val

    def FOR_ITER(self, instr):
        iterator = self.stack.pop()
        backup_iter_idx = None

        start = self.indexof(instr)
        end = self.indexof(instr.jump_to)
        for i in range(start, end):
            if self._instructions[i].opname == "RETURN_VALUE":
                raise FallbackError("Found RETURN_VALUE in for loop body.")

        self._graph.add_global_guarded_variable(iterator)

        try:
            if not isinstance(iterator, SequenceIterVariable):
                raise BreakGraphError()

            backup_iter_idx = iterator.idx

            self._inline_call_for_loop(iterator, instr)
            self._lasti = self.indexof(instr.jump_to)
        except BreakGraphError as e:
            log(3, f"{e}")
            if backup_iter_idx:
                iterator.idx = backup_iter_idx
            self._graph.remove_global_guarded_variable(iterator)
            self._break_graph_in_for_loop(iterator, instr)
            return Stop(state="BreakGraph")

    def RETURN_VALUE(self, instr: Instruction):
        assert (
            len(self.stack) == 1
        ), f"Stack must have one element, but get {len(self.stack)} elements."
        ret_val = self.stack.pop()
        self._graph.start_compile(ret_val)
        self._graph.pycode_gen.gen_return()
        self.new_code = self._graph.pycode_gen.gen_pycode()
        self.guard_fn = self._graph.guard_fn
        return Stop(state="Return")
