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

import collections
import dis
import inspect
import operator
import types
from typing import Callable, List, Optional, Tuple

from ...utils import (
    BreakGraphError,
    InnerError,
    Singleton,
    UnsupportError,
    is_strict_mode,
    log,
    log_do,
)
from ..instruction_utils.instruction_utils import Instruction, get_instructions
from .function_graph import FunctionGraph
from .guard import Guard
from .instr_flag import FORMAT_VALUE_FLAG as FV
from .instr_flag import MAKE_FUNCTION_FLAG as MF
from .pycode_generator import PyCodeGen
from .tracker import (
    BuiltinTracker,
    DummyTracker,
    GetItemTracker,
    GlobalTracker,
    LocalTracker,
)
from .variables import (
    CallableVariable,
    ConstantVariable,
    ConstTracker,
    ContainerVariable,
    DictIterVariable,
    DictVariable,
    IterVariable,
    ListVariable,
    ObjectVariable,
    SequenceIterVariable,
    TensorIterVariable,
    TensorVariable,
    TupleVariable,
    UserDefinedFunctionVariable,
    UserDefinedIterVariable,
    VariableBase,
    VariableFactory,
)

CustomCode = collections.namedtuple(
    "CustomCode", ["code", "disable_eval_frame"]
)


GuardedFunction = Tuple[types.CodeType, Guard]
GuardedFunctions = List[GuardedFunction]
CacheGetter = Callable[
    [types.FrameType, GuardedFunctions], Optional[CustomCode]
]
dummy_guard: Guard = lambda frame: True

SUPPORT_COMPARE_OP = {
    ">": operator.gt,
    "<": operator.lt,
    ">=": operator.ge,
    "<=": operator.le,
    "==": lambda x, y: VariableFactory.from_value(
        x.value == y.value, None, tracker=DummyTracker([x, y])
    ),
    "!=": lambda x, y: VariableFactory.from_value(
        x.value != y.value, None, tracker=DummyTracker([x, y])
    ),
    "is not": lambda x, y: VariableFactory.from_value(
        x.value is not y.value, None, tracker=DummyTracker([x, y])
    ),
    "is": lambda x, y: VariableFactory.from_value(
        x.value is y.value, None, tracker=DummyTracker([x, y])
    ),
}


class Stop:
    pass


@Singleton
class InstructionTranslatorCache:
    cache: dict[types.CodeType, tuple[CacheGetter, GuardedFunctions]]
    translate_count: int

    def __init__(self):
        self.cache = {}
        self.translate_count = 0

    def clear(self):
        self.cache.clear()
        self.translate_count = 0

    def __call__(self, frame) -> CustomCode | None:
        code: types.CodeType = frame.f_code
        if code not in self.cache:
            cache_getter, (new_code, guard_fn) = self.translate(frame)
            self.cache[code] = (cache_getter, [(new_code, guard_fn)])
            if cache_getter == self.skip:
                return None
            return CustomCode(new_code, False)
        cache_getter, guarded_fns = self.cache[code]
        return cache_getter(frame, guarded_fns)

    def lookup(
        self, frame: types.FrameType, guarded_fns: GuardedFunctions
    ) -> CustomCode | None:
        for code, guard_fn in guarded_fns:
            try:
                if guard_fn(frame):
                    log(3, "[Cache]: Cache hit\n")
                    return CustomCode(code, True)
            except Exception as e:
                log(3, f"[Cache]: Guard function error: {e}\n")
                continue
        cache_getter, (new_code, guard_fn) = self.translate(frame)
        guarded_fns.append((new_code, guard_fn))
        return CustomCode(new_code, False)

    def skip(
        self, frame: types.FrameType, guarded_fns: GuardedFunctions
    ) -> CustomCode | None:
        log(3, f"[Cache]: Skip frame {frame.f_code.co_name}\n")
        return None

    def translate(
        self, frame: types.FrameType
    ) -> tuple[CacheGetter, GuardedFunction]:
        code: types.CodeType = frame.f_code
        log(3, "[Cache]: Cache miss\n")
        self.translate_count += 1

        result = start_translate(frame)
        if result is None:
            return self.skip, (code, dummy_guard)

        new_code, guard_fn = result
        return self.lookup, (new_code, guard_fn)


def start_translate(frame) -> GuardedFunction | None:
    simulator = OpcodeExecutor(frame)
    try:
        new_code, guard_fn = simulator.transform()
        log_do(3, lambda: dis.dis(new_code))
        return new_code, guard_fn
    except InnerError as e:
        raise
    # TODO(0x45f): handle BreakGraphError to trigger fallback
    except (UnsupportError, BreakGraphError) as e:
        if is_strict_mode():
            raise
        log(
            2,
            f"Unsupport Frame is {frame.f_code}, error message is: {str(e)}\n",
        )
        return None
    except Exception as e:
        raise


def tos_op_wrapper(fn):
    nargs = len(inspect.signature(fn).parameters)

    def inner(self: OpcodeExecutorBase, instr: Instruction):
        args = self.pop_n(nargs)
        self.push(fn(*args))

    return inner


def breakoff_graph_with_jump(normal_jump):
    """breakoff graph when meet jump."""

    def jump_instruction_with_fallback(self: OpcodeExecutor, instr):
        result = self.peek()
        if isinstance(result, TensorVariable):
            self.pop()
            # fallback when in OpcodeExecutor
            # raise error in OpcodeInlineExecutor
            self._fallback_in_jump(result, instr)
            return Stop()
        else:
            return normal_jump(self, instr)

    return jump_instruction_with_fallback


class OpcodeExecutorBase:
    def __init__(self, code: types.CodeType, graph: FunctionGraph):
        # fake env for run, new env should be gened by PyCodeGen
        self._stack: list[VariableBase] = []
        self._co_consts = []
        self._locals = {}
        self._globals = {}
        self._builtins = {}
        self._lasti = 0  # idx of instruction list
        self._code = code
        self._instructions = get_instructions(self._code)
        self._graph = graph
        self.new_code = None
        self.guard_fn = None
        self._prepare_virtual_env()

    def _prepare_virtual_env(self):
        raise NotImplementedError("Please inplement virtual_env.")

    def transform(self):
        raise NotImplementedError()

    def run(self):
        log(3, f"start execute opcode: {self._code}\n")
        self._lasti = 0
        while True:
            if self._lasti >= len(self._instructions):
                raise InnerError("lasti out of range, InnerError.")
            cur_instr = self._instructions[self._lasti]
            self._lasti += 1
            is_stop = self.step(cur_instr)
            if is_stop:
                break

    def step(self, instr):
        if not hasattr(self, instr.opname):
            raise UnsupportError(f"opcode: {instr.opname} is not supported.")
        log(3, f"[TraceExecution]: {instr.opname}, stack is {self._stack}\n")
        return getattr(self, instr.opname)(instr)  # run single step.

    def indexof(self, instr):
        return self._instructions.index(instr)

    def pop(self) -> VariableBase:
        return self._stack.pop()

    def peek(self) -> VariableBase:
        return self._stack[-1]

    def peek_n(self, n) -> list[VariableBase]:
        return self._stack[-n:]

    def pop_n(self, n: int) -> list[VariableBase]:
        if n == 0:
            return []
        retval = self._stack[-n:]
        self._stack[-n:] = []
        return retval

    def push(self, val: VariableBase):
        self._stack.append(val)

    # unary operators
    UNARY_POSITIVE = tos_op_wrapper(operator.pos)
    UNARY_NEGATIVE = tos_op_wrapper(operator.neg)
    # UNARY_NOT = tos_op_wrapper(operator.not_)
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

    # inplace operators
    # paddle variable do not have inplace operators. For example when call `y **= x`, will call var.__pow__
    INPLACE_POWER = tos_op_wrapper(operator.ipow)
    INPLACE_MULTIPLY = tos_op_wrapper(operator.imul)
    INPLACE_MATRIX_MULTIPLY = tos_op_wrapper(operator.imatmul)
    INPLACE_FLOOR_DIVIDE = tos_op_wrapper(operator.ifloordiv)
    INPLACE_TRUE_DIVIDE = tos_op_wrapper(operator.itruediv)
    INPLACE_MODULO = tos_op_wrapper(operator.imod)
    INPLACE_ADD = tos_op_wrapper(operator.iadd)
    INPLACE_SUBTRACT = tos_op_wrapper(operator.isub)
    INPLACE_LSHIFT = tos_op_wrapper(operator.ilshift)
    INPLACE_RSHIFT = tos_op_wrapper(operator.irshift)
    INPLACE_AND = tos_op_wrapper(operator.iand)
    INPLACE_OR = tos_op_wrapper(operator.ior)
    INPLACE_XOR = tos_op_wrapper(operator.ixor)

    def LOAD_ATTR(self, instr):
        attr_name = instr.argval
        obj = self.pop()
        self.push(getattr(obj, attr_name))

    def LOAD_FAST(self, instr):
        varname = instr.argval
        var = self._locals[varname]
        self.push(var)

    def LOAD_METHOD(self, instr):
        method_name = instr.argval
        obj = self.pop()
        method = getattr(obj, method_name)
        self.push(method)

    def STORE_FAST(self, instr):
        """
        TODO: side effect may happen
        """
        var = self.pop()
        self._locals[instr.argval] = var

    def LOAD_GLOBAL(self, instr):
        name = instr.argval
        if name in self._globals.keys():
            value = self._globals[name]
        else:
            value = self._builtins[name]
        self.push(value)

    def LOAD_CONST(self, instr):
        var = self._co_consts[instr.arg]
        self.push(var)

    def BINARY_SUBSCR(self, instr):
        key = self.pop()
        container = self.pop()
        assert isinstance(key, VariableBase)
        self._graph.add_global_guarded_variable(key)
        self.push(container[key.value])

    def STORE_SUBSCR(self, instr):
        key = self.pop()
        container = self.pop()
        value = self.pop()
        assert isinstance(key, VariableBase)
        self._graph.add_global_guarded_variable(key)
        container[key.value] = value

    def CALL_FUNCTION(self, instr):
        n_args = instr.arg
        assert n_args <= len(self._stack)
        args = self.pop_n(n_args)
        kwargs = {}
        fn = self.pop()
        if not isinstance(fn, CallableVariable):
            raise UnsupportError(f"CALL_FUNCTION: {fn} is not callable")
        ret = fn(*args, **kwargs)
        self.push(ret)

    def CALL_FUNCTION_KW(self, instr):
        n_args = instr.arg
        assert n_args + 2 <= len(self._stack)

        kwargs_keys = self.pop()
        assert isinstance(kwargs_keys, TupleVariable)
        assert len(kwargs_keys) > 0
        kwargs_keys = [
            x.value if isinstance(x, VariableBase) else x
            for x in kwargs_keys.value
        ]

        # split arg_list to args and kwargs
        arg_list = self.pop_n(n_args)
        args = arg_list[0 : -len(kwargs_keys)]
        kwargs_values = arg_list[-len(kwargs_keys) :]
        kwargs = dict(zip(kwargs_keys, kwargs_values))

        fn = self.pop()
        if not isinstance(fn, CallableVariable):
            raise UnsupportError(f"CALL_FUNCTION_KW: {fn} is not callable.")
        ret = fn(*args, **kwargs)
        self.push(ret)

    def CALL_FUNCTION_EX(self, instr):
        flag = instr.arg
        if flag & 0x01:  # has kwargs
            kwargs_variable = self.pop()
            assert isinstance(kwargs_variable, DictVariable)
            kwargs = kwargs_variable.get_wrapped_items()
        else:
            kwargs = {}

        args_variable = self.pop()
        assert isinstance(args_variable, TupleVariable)
        args = args_variable.get_wrapped_items()

        fn = self.pop()
        if not isinstance(fn, CallableVariable):
            raise UnsupportError(f"CALL_FUNCTION_EX: {fn} is not callable.")
        ret = fn(*args, **kwargs)
        self.push(ret)

    def CALL_METHOD(self, instr):
        n_args = instr.argval
        assert n_args <= len(self._stack)
        args = self.pop_n(n_args)
        method = self.pop()
        if not isinstance(method, CallableVariable):
            raise UnsupportError(f"CALL METHOD: {method} is not callable.")
        ret = method(*args)
        self.push(ret)

    def COMPARE_OP(self, instr):
        op = instr.argval
        if op in SUPPORT_COMPARE_OP:
            right, left = self.pop(), self.pop()
            self.push(SUPPORT_COMPARE_OP[op](left, right))
            return
        else:
            raise UnsupportError(
                f"{instr} is not support. may be not a supported compare op."
            )

    @breakoff_graph_with_jump
    def JUMP_IF_FALSE_OR_POP(self, instr):
        pred_obj = self.peek()
        if isinstance(pred_obj, (ConstantVariable, ContainerVariable)):
            self._graph.add_global_guarded_variable(pred_obj)
            is_jump = not bool(pred_obj)
            if is_jump:
                self._lasti = self.indexof(instr.jump_to)
            else:
                self.pop()
            return
        raise UnsupportError(
            "Currently don't support predicate a non-const / non-tensor obj."
        )

    @breakoff_graph_with_jump
    def JUMP_IF_TRUE_OR_POP(self, instr):
        pred_obj = self.peek()
        if isinstance(pred_obj, (ConstantVariable, ContainerVariable)):
            self._graph.add_global_guarded_variable(pred_obj)
            is_jump = bool(pred_obj)
            if is_jump:
                self._lasti = self.indexof(instr.jump_to)
            else:
                self.pop()
            return
        raise UnsupportError(
            "Currently don't support predicate a non-const / non-tensor obj."
        )

    @breakoff_graph_with_jump
    def POP_JUMP_IF_FALSE(self, instr):
        pred_obj = self.pop()
        if isinstance(pred_obj, (ConstantVariable, ContainerVariable)):
            self._graph.add_global_guarded_variable(pred_obj)
            is_jump = not bool(pred_obj)
            if is_jump:
                self._lasti = self.indexof(instr.jump_to)
            return
        raise UnsupportError(
            "Currently don't support predicate a non-const / non-tensor obj."
        )

    @breakoff_graph_with_jump
    def POP_JUMP_IF_TRUE(self, instr):
        pred_obj = self.pop()
        if isinstance(pred_obj, (ConstantVariable, ContainerVariable)):
            self._graph.add_global_guarded_variable(pred_obj)
            is_jump = bool(pred_obj)
            if is_jump:
                self._lasti = self.indexof(instr.jump_to)
            return
        raise UnsupportError(
            "Currently don't support predicate a non-const / non-tensor obj."
        )

    def _fallback_in_jump(self, result, instr):
        raise NotImplementedError()

    def JUMP_FORWARD(self, instr):
        self._lasti = self.indexof(instr.jump_to)

    def JUMP_ABSOLUTE(self, instr):
        self._lasti = self.indexof(instr.jump_to)

    def RETURN_VALUE(self, instr):
        assert (
            len(self._stack) == 1
        ), f"Stack must have one element, but get {len(self._stack)} elements."
        ret_val = self.pop()
        self._graph.start_compile(ret_val)
        self._graph.pycode_gen.gen_return()
        self.new_code = self._graph.pycode_gen.gen_pycode()
        self.guard_fn = self._graph.guard_fn
        return Stop()

    def BUILD_LIST(self, instr):
        list_size = instr.arg
        assert list_size <= len(
            self._stack
        ), f"OpExecutor want BUILD_LIST with size {list_size}, but current stack do not have enough elems."
        val_list = self.pop_n(list_size)
        self.push(
            VariableFactory.from_value(
                val_list, graph=self._graph, tracker=DummyTracker(val_list)
            )
        )

    def BUILD_TUPLE(self, instr):
        tuple_size = instr.arg
        assert tuple_size <= len(
            self._stack
        ), f"OpExecutor want BUILD_TUPLE with size {tuple_size}, but current stack do not have enough elems."
        val_tuple = self.pop_n(tuple_size)
        self.push(
            VariableFactory.from_value(
                tuple(val_tuple),
                graph=self._graph,
                tracker=DummyTracker(val_tuple),
            )
        )

    def BUILD_MAP(self, instr):
        map_size = instr.arg
        built_map = {}
        assert map_size * 2 <= len(
            self._stack
        ), f"OpExecutor want BUILD_MAP with size {map_size} * 2, but current stack do not have enough elems."
        val_for_dict = self.pop_n(map_size * 2)
        keys = val_for_dict[::2]
        values = val_for_dict[1::2]
        self.push(self.build_map(keys, values))

    def BUILD_CONST_KEY_MAP(self, instr):
        map_size = instr.arg
        assert map_size + 1 <= len(
            self._stack
        ), f"OpExecutor want BUILD_CONST_KEY_MAP with size {map_size} + 1, but current stack do not have enough elems."
        keys = self.pop().get_items()
        assert len(keys) == map_size
        values = self.pop_n(map_size)
        self.push(self.build_map(keys, values))

    def build_map(
        self, keys: list[VariableBase], values: list[VariableBase]
    ) -> VariableBase:
        built_map = {}
        for key, value in zip(keys, values):
            assert isinstance(key, VariableBase)
            # Add key to global guarded variable to avoid missing the key guard
            self._graph.add_global_guarded_variable(key)
            key = key.value
            built_map[key] = value
        return DictVariable(
            built_map,
            graph=self._graph,
            tracker=DummyTracker(keys + values),
        )

    def _rot_top_n(self, n):
        # a1 a2 a3 ... an  <- TOS
        # the stack changes to
        # an a1 a2 a3 an-1 <- TOS
        assert (
            len(self._stack) >= n
        ), f"There are not enough elements on the stack. {n} is needed."
        top = self.pop()
        self._stack[-(n - 1) : -(n - 1)] = [top]

    def POP_TOP(self, instr):
        self.pop()

    def ROT_TWO(self, instr):
        self._rot_top_n(2)

    def ROT_THREE(self, instr):
        self._rot_top_n(3)

    def ROT_FOUR(self, instr):
        self._rot_top_n(4)

    def UNPACK_SEQUENCE(self, instr):
        sequence = self.pop()

        '''
            TODO: To unpack iterator
            To unpack is easy, just like:
                seq = tuple(sequence.value)

            But what is the `source` when iterator returned a value ?
        '''
        if isinstance(sequence, TensorVariable):
            # TODO: If need to unpack a Tensor, should have different logic.
            raise NotImplementedError("Unpack a iterator is not implemented.")
        elif isinstance(sequence, (ListVariable, TupleVariable)):
            seq = sequence.value
        else:
            raise NotImplementedError(f"Unpack {sequence} is not implemented.")

        assert (
            len(seq) == instr.arg
        ), f"Want unpack {seq} to {instr.arg}, but the len is {len(seq)}."

        for i in range(instr.arg - 1, -1, -1):
            self.push(
                VariableFactory.from_value(
                    seq[i],
                    graph=self._graph,
                    tracker=GetItemTracker(sequence, i),
                )
            )

    def BUILD_STRING(self, instr):
        count = instr.arg
        assert count <= len(
            self._stack
        ), f"OpExecutor want BUILD_STRING with size {count}, but current stack do not have enough elems."
        str_list = self.pop_n(count)
        new_str = ''
        for s in str_list:
            assert isinstance(s.value, str)
            new_str += s.value
        self.push(ConstantVariable.wrap_literal(new_str))

    def FORMAT_VALUE(self, instr):

        flag = instr.arg
        which_conversion = flag & FV.FVC_MASK
        have_fmt_spec = bool((flag & FV.FVS_MASK) == FV.FVS_HAVE_SPEC)

        fmt_spec = self.pop().value if have_fmt_spec else ""
        value = self.pop()

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

        # different type will lead to different Tracker, so call self.push in different branch
        if isinstance(value, ConstantVariable):
            result = value.value
            if convert_fn is not None:
                result = getattr(result, convert_fn)(result)

            if not isinstance(result, str) or fmt_spec != "":
                result = format(result, fmt_spec)

            self.push(
                VariableFactory.from_value(
                    result, self._graph, DummyTracker([value])
                )
            )
        else:
            raise UnsupportError(f"Do not support format {type(value)} now")

    def build_seq_unpack(self, instr):
        oparg = instr.arg
        assert oparg <= len(self._stack)
        unpack_values = self.pop_n(oparg)

        retval = []
        for item in unpack_values:
            assert isinstance(item, (TupleVariable, ListVariable))
            retval.extend(item.get_wrapped_items())

        if instr.opname in {
            "BUILD_TUPLE_UNPACK_WITH_CALL",
            "BUILD_TUPLE_UNPACK",
        }:
            retval = tuple(retval)

        self.push(
            VariableFactory.from_value(
                retval, self._graph, DummyTracker(unpack_values)
            )
        )

    def BUILD_TUPLE_UNPACK_WITH_CALL(self, instr):
        self.build_seq_unpack(instr)

    def BUILD_TUPLE_UNPACK(self, instr):
        self.build_seq_unpack(instr)

    def BUILD_LIST_UNPACK(self, instr):
        self.build_seq_unpack(instr)

    def BUILD_MAP_UNPACK(self, instr):
        oparg = instr.arg
        assert oparg <= len(self._stack)
        unpack_values = self.pop_n(oparg)

        retval = {}
        for item in unpack_values:
            assert isinstance(item.value, dict)
            retval.update(item.get_wrapped_items())

        self.push(
            VariableFactory.from_value(
                retval, self._graph, DummyTracker(unpack_values)
            )
        )

    def BUILD_MAP_UNPACK_WITH_CALL(self, instr):
        oparg = instr.arg
        assert oparg <= len(self._stack)
        unpack_values = self.pop_n(oparg)

        retval = {}
        for item in unpack_values:
            assert isinstance(item.value, dict)
            wrapped_item = item.get_wrapped_items()
            if wrapped_item.items() & retval.items():
                raise InnerError(
                    "BUILD_MAP_UNPACK_WITH_CALL found repeated key."
                )
            retval.update(wrapped_item)

        self.push(
            VariableFactory.from_value(
                retval, self._graph, DummyTracker(unpack_values)
            )
        )

    def MAKE_FUNCTION(self, instr):
        fn_name = self.pop()
        codeobj = self.pop()
        global_dict = self._globals

        related_list = [fn_name, codeobj]

        flag = instr.arg
        if flag & MF.MF_HAS_CLOSURE:
            # closure should be a tuple of Variables
            closure_variable = self.pop()
            assert isinstance(closure_variable, TupleVariable)
            related_list.append(closure_variable)
            closure = tuple(closure_variable.get_wrapped_items())
        else:
            closure = ()

        if flag & MF.MF_HAS_ANNOTATION:
            # can not set annotation in python env, skip it
            related_list.append(self.pop())

        if flag & MF.MF_HAS_KWDEFAULTS:
            raise UnsupportError(
                "Found need func_kwdefaults when MAKE_FUNCTION."
            )

        if flag & MF.MF_HAS_DEFAULTS:
            '''
            default_args should have tracker too, like:

            def f(x):
                def g(z=x):
                    pass
            '''
            default_args_variable = self.pop()
            assert isinstance(default_args_variable, TupleVariable)
            related_list.append(default_args_variable)
            default_args = tuple(default_args_variable.get_wrapped_items())
        else:
            default_args = ()

        new_fn = types.FunctionType(
            codeobj.value, global_dict, fn_name.value, default_args, closure
        )

        self.push(
            UserDefinedFunctionVariable(
                new_fn, self._graph, DummyTracker(related_list)
            )
        )

    def BUILD_SLICE(self, instr):
        if instr.arg == 3:
            step = self.pop()
        else:
            step = None
        stop = self.pop()
        start = self.pop()

        related_list = [start, stop, step] if step else [start, stop]

        slice_ = slice(*(x.value for x in related_list))

        self.push(
            VariableFactory.from_value(
                slice_, self._graph, DummyTracker(related_list)
            )
        )

    def DUP_TOP(self, instr):
        self.push(self.peek())

    def DUP_TOP_TWO(self, instr):
        for ref in self.peek_n(2):
            self.push(ref)

    def NOP(self, instr):
        pass

    def GET_ITER(self, instr):
        iterator = self.pop()
        if isinstance(iterator, IterVariable):
            return self.push(iterator)

        if isinstance(iterator, (ListVariable, TupleVariable)):
            self.push(
                SequenceIterVariable(
                    iterator, self._graph, DummyTracker([iterator])
                )
            )
        elif isinstance(iterator, DictVariable):
            self.push(
                DictIterVariable(
                    iterator, self._graph, DummyTracker([iterator])
                )
            )
        elif isinstance(iterator, TensorVariable):
            self.push(
                TensorIterVariable(
                    iterator, self._graph, DummyTracker([iterator])
                )
            )
        else:
            self.push(
                UserDefinedIterVariable(
                    iterator, self._graph, DummyTracker([iterator])
                )
            )

    def FOR_ITER(self, instr):
        iterator = self.pop()
        assert isinstance(iterator, IterVariable)

        # simplely get next
        if isinstance(iterator, (SequenceIterVariable, DictIterVariable)):
            try:
                val, next_iterator = iterator.next()
                self.push(
                    next_iterator
                )  # need a new iterator to replace the old one
                self.push(val)
            except StopIteration:
                self._lasti = self.indexof(instr.jump_to)

        # TODO need support TensorIterVariable.next

        else:
            self._fallback_in_for_loop(iterator, instr)
            return Stop()


class OpcodeExecutor(OpcodeExecutorBase):
    def __init__(self, frame):
        graph = FunctionGraph(frame)
        self._frame = frame
        super().__init__(frame.f_code, graph)

    def _prepare_virtual_env(self):
        for name, value in self._frame.f_locals.items():
            self._locals[name] = VariableFactory.from_value(
                value, self._graph, LocalTracker(name)
            )

        for name, value in self._frame.f_globals.items():
            self._globals[name] = VariableFactory.from_value(
                value, self._graph, GlobalTracker(name)
            )

        for name, value in self._frame.f_builtins.items():
            self._builtins[name] = VariableFactory.from_value(
                value, self._graph, BuiltinTracker(name)
            )

        for value in self._code.co_consts:
            self._co_consts.append(
                VariableFactory.from_value(
                    value, self._graph, ConstTracker(value)
                )
            )

    def _create_resume_fn(self, index):
        pycode_gen = PyCodeGen(self._frame)
        fn, inputs = pycode_gen.gen_resume_fn_at(index)
        return fn, inputs

    def _fallback_in_jump(self, result, instr):
        if_fn, if_inputs = self._create_resume_fn(self.indexof(instr) + 1)
        else_fn, else_inputs = self._create_resume_fn(
            self.indexof(instr.jump_to)
        )
        inputs_name = if_inputs | else_inputs
        inputs_var = [
            self._locals[name]
            for name in inputs_name
            if self._locals[name] is not result
        ]
        ret_vars = [
            result,
        ] + inputs_var
        self._graph.start_compile(*ret_vars)
        for _ in inputs_var:
            self._graph.pycode_gen.gen_pop_top()

        if if_fn is not None:
            self._graph.pycode_gen.gen_load_object(
                if_fn, if_fn.__code__.co_name
            )
            insert_index = len(self._graph.pycode_gen._instructions) - 1
            for name in if_inputs:
                self._locals[name].reconstruct(self._graph.pycode_gen)
            self._graph.pycode_gen.gen_call_function(
                argc=if_fn.__code__.co_argcount
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
            for name in else_inputs:
                self._locals[name].reconstruct(self._graph.pycode_gen)
            self._graph.pycode_gen.gen_call_function(
                argc=else_fn.__code__.co_argcount
            )
            self._graph.pycode_gen.gen_return()
        else:
            self._graph.pycode_gen.gen_return()
            jump_to = self._graph.pycode_gen._instructions[-1]

        self._graph.pycode_gen._insert_instr(
            insert_index, instr.opname, jump_to=jump_to
        )

        self.new_code = self._graph.pycode_gen.gen_pycode()
        self.guard_fn = self._graph.guard_fn

    def transform(self):
        self.run()
        if self.new_code is None:
            raise InnerError("OpExecutor return a empty new_code.")
        return self.new_code, self.guard_fn

    def _create_loop_body_fn(self, start, end):
        pycode_gen = PyCodeGen(self._frame)
        fn, inputs = pycode_gen.gen_loop_body_fn_between(start, end)
        return fn, inputs

    def _fallback_in_for_loop(self, iterator, instr):
        '''
        instr: the FOR_ITER opcode

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
        '''
        unpack_instr_idx = self.indexof(instr) + 1
        curent_stack = 1

        while True:
            if unpack_instr_idx >= len(self._instructions):
                raise InnerError("Can not balance stack in loop body.")
            cur_instr = self._instructions[unpack_instr_idx]
            # do not consider jump instr
            stack_effect = dis.stack_effect(
                cur_instr.opcode, cur_instr.arg, jump=False
            )
            curent_stack += stack_effect
            unpack_instr_idx += 1
            if curent_stack == 0:
                break

        loop_body, loop_inputs = self._create_loop_body_fn(
            unpack_instr_idx, self.indexof(instr.jump_to)
        )

        after_loop_fn, fn_inputs = self._create_resume_fn(
            self.indexof(instr.jump_to)
        )

        # 1. part before for-loop
        inputs_var = [
            self._locals[name] for name in loop_inputs if name in self._locals
        ]
        self._graph.start_compile(*inputs_var)

        for _ in inputs_var:
            self._graph.pycode_gen.gen_pop_top()

        # 2. load iterator to stack
        iterator.reconstruct(self._graph.pycode_gen)

        # 3. gen FOR_ITER and unpack data
        self._graph.pycode_gen.extend_instrs(
            self._instructions[self.indexof(instr) : unpack_instr_idx]
        )

        # 4. call loop body
        self._graph.pycode_gen.gen_load_object(
            loop_body, loop_body.__code__.co_name
        )

        def update_locals(name, variable):
            self._locals[name] = variable
            return variable

        for name in loop_inputs:
            if name in self._locals:
                self._locals[name].reconstruct(self._graph.pycode_gen)
            elif name in self._globals:
                self._globals[name].reconstruct(self._graph.pycode_gen)
            elif name in self._builtins:
                self._builtins[name].reconstruct(self._graph.pycode_gen)
            else:
                variable = update_locals(
                    name, ObjectVariable(None, self._graph, LocalTracker(name))
                )
                variable.reconstruct(self._graph.pycode_gen)

        self._graph.pycode_gen.gen_call_function(
            argc=loop_body.__code__.co_argcount
        )

        # 5. unpack and store
        self._graph.pycode_gen.gen_unpack_sequence(len(loop_inputs))
        for name in loop_inputs:
            self._graph.pycode_gen.gen_store_fast(
                name
            )  # TODO: need check data scope with globals, builtins

        # 6. add JUMP_ABSOLUTE
        self._graph.pycode_gen.gen_jump_abs(instr)

        # 7. call after_loop_fn
        self._graph.pycode_gen.gen_load_object(
            after_loop_fn, after_loop_fn.__code__.co_name
        )

        for name in fn_inputs:
            if name in self._locals:
                self._locals[name].reconstruct(self._graph.pycode_gen)
            elif name in self._globals:
                self._globals[name].reconstruct(self._graph.pycode_gen)
            elif name in self._builtins:
                self._builtins[name].reconstruct(self._graph.pycode_gen)

        self._graph.pycode_gen.gen_call_function(
            argc=after_loop_fn.__code__.co_argcount
        )

        self._graph.pycode_gen.gen_return()
        self.new_code = self._graph.pycode_gen.gen_pycode()
        self.guard_fn = self._graph.guard_fn
