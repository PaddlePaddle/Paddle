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

from typing import TYPE_CHECKING

import paddle
from paddle.utils import to_sequence

from ..utils import InnerError, log_do, map_if, map_if_extend
from .statement_ir import SIRRuntimeCache, Symbol

if TYPE_CHECKING:
    from .statement_ir import Statement, StatementIR
    from .symbolic_context import SymbolicTraceContext


def replace_symbol(
    values: list[Symbol] | list[object], state: dict[str, Symbol]
):
    """
    Replaces Symbol objects with their corresponding values.

    Args:
        values: A list of values that may contain Symbol objects.
        state: A dict mapping Symbol names to their corresponding values.

    Returns:
        A new list with Symbol objects replaced by their corresponding values in the state dict.
    """
    # deal with list / map etc.
    values = map_if_extend(
        values,
        pred=lambda x: isinstance(x, Symbol),
        true_fn=lambda x: state[x.name],
        false_fn=lambda x: x,
    )
    return values


def _append_opstack_between(start, end, stack):
    # The range is [start, end)
    from paddle.framework import core

    op_maker = core.op_proto_and_checker_maker
    callstack_attr_name = op_maker.kOpCreationCallstackAttrName()
    for op in for_each_ops_between(start, end):
        if paddle.framework.use_pir_api():
            op.callstack = stack
        else:
            # NOTE(xiongkun): we don't sync for speed. careful!!
            op._set_attr(callstack_attr_name, stack)


def for_each_ops_between(start, end):
    # NOTE(xiongkun): we don't sync for speed. careful!!
    # [start, end)
    program = paddle.static.default_main_program()
    ops = program.global_block().ops[start:end]
    yield from ops


def opnum_in_program():
    # NOTE(xiongkun): we don't sync for speed. careful!!
    program = paddle.static.default_main_program()
    return len(program.global_block().ops)


class Interpreter:
    """
    Interpreter is used to interpret and execute SIR.
    """

    def __init__(self, symbolic_context: SymbolicTraceContext):
        self._context = symbolic_context

    def get_sir(self, name: str) -> StatementIR:
        """
        Returns the StatementIR object by given name.

        Args:
            name: The name of the StatementIR.

        Returns:
            The StatementIR object with the given name.
        """
        return self._context.get_sir(name)

    def run_sir(self, name: str, state: dict[str, Symbol]):
        """
        Runs the StatementIR with the given name using the provided state.

        Args:
            name: The name of the given StatementIR to run.
            state: A dict mapping Symbol names to their corresponding values.

        Returns:
            A list of the Symbol of the StatementIR after execution.
        """
        SIR = self.get_sir(name)
        for stmt in SIR.statements:
            stmt: Statement
            before_stmt_opnum = opnum_in_program()
            inputs = replace_symbol(stmt.inputs, state)
            outs = getattr(self, stmt.type)(stmt, inputs)

            def _set(v, s):
                state[s.name] = v

            if len(to_sequence(outs)) != len(to_sequence(stmt.outputs)):
                raise InnerError("Number output mismatch, some error happen.")

            log_do(
                3,
                lambda: _append_opstack_between(
                    before_stmt_opnum, opnum_in_program() + 1, stmt.stmt_stack
                ),
            )

            map_if(
                outs,
                stmt.outputs,
                pred=lambda v, s: isinstance(s, Symbol),
                true_fn=lambda v, s: _set(v, s),
                false_fn=lambda v, s: None,
            )
        # fetch outputs
        return replace_symbol(SIR.outputs, state)

    def call(self, stmt: Statement, inputs):
        SIR = self.get_sir(stmt.sir_name)
        state = prepare_state(SIR, inputs)
        return self.run_sir(stmt.sir_name, state)

    def api(self, stmt, inputs):
        args, kwargs = inputs
        return stmt.api(*args, **kwargs)

    def method(self, stmt, inputs):
        args, kwargs = inputs
        var = args[0]
        return getattr(var, stmt.method)(*args[1:], **kwargs)

    def layer(self, stmt, inputs):
        args, kwargs = inputs
        layer = stmt.layer()
        assert layer is not None, "SIR bound layer is None."
        return layer(*args, **kwargs)

    def AST(self, stmt, inputs):
        args, kwargs = inputs
        return stmt.converted_func(*args, **kwargs)


def compile_sir(context: SymbolicTraceContext, name: str):
    """
    Compile a SIR to a new function

    Args:
        context: The context to compile
        name: The name of the sir to compile

    """

    @paddle.jit.not_to_static
    def wrapper(args):
        """
        This function will be decorated by paddle.to_static.
        so the args is variables, not eager tensors.
        """
        interpreter = Interpreter(context)
        SIR = interpreter.get_sir(name)
        state = prepare_state(SIR, args)
        return interpreter.run_sir(name, state)

    return wrapper


def prepare_state(SIR, inputs):
    state = {}

    # update free vars if exists
    if SIRRuntimeCache().has_key(SIR.name):
        free_var_seeker = SIRRuntimeCache().get_free_vars(SIR.name)
        if free_var_seeker:
            state = free_var_seeker()

    # bind inputs
    for sir_inp, inp in zip(SIR.inputs, inputs):
        state[sir_inp.name] = inp

    return state
