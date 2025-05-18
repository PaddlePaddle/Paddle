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

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable

from ..utils import log
from .compile_cache import CompileSIRCache
from .statement_ir import (
    ApiStatement,
    ASTStatement,
    CallStatement,
    LayerStatement,
    MethodStatement,
    StatementContext,
    StatementIR,
    StatementIRFactory,
    Symbol,
)

if TYPE_CHECKING:
    from paddle.static import InputSpec


class StatementIRBuilder:
    """
    A class to build a StatementIR.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """
        Reset the context.
        """

        self.statement_factory = StatementIRFactory()
        self._current_statement_ctxs = []
        self._current_sir: StatementIR = self.statement_factory.create()

    @property
    def current_sir(self) -> StatementIR:
        """
        Get the current SIR.
        """
        return self._current_sir

    def replace_current_sir(self, sir: StatementIR):
        """
        Replace the current SIR with a new SIR.
        """
        self._current_sir = sir
        self.statement_factory.update(sir)

    def call_SIR(self, sirname, inputs, outputs, stacks):
        """
        Call a SIR, which is a subgraph.
        """

        stmt = CallStatement(
            sirname, inputs, outputs, list(self._current_statement_ctxs), stacks
        )
        self.current_sir.add_statement(stmt)

    def call_API(self, api, inputs, outputs, stacks):
        """
        Call a paddle api.
        """
        assert callable(api), "call_API must receive a paddle api."
        stmt = ApiStatement(
            api, inputs, outputs, list(self._current_statement_ctxs), stacks
        )
        self.current_sir.add_statement(stmt)

    def call_METHOD(self, method_name, inputs, outputs, stacks):
        """
        Call a method of a api. The API here can be python or Paddle
        """
        assert isinstance(
            method_name, str
        ), "call_METHOD must method api name. string."
        assert isinstance(
            inputs[0][0], Symbol
        ), "call_METHOD first argument must be Symbol Variable."
        stmt = MethodStatement(
            method_name,
            inputs,
            outputs,
            list(self._current_statement_ctxs),
            stacks,
        )
        self.current_sir.add_statement(stmt)

    def call_LAYER(self, layer, inputs, outputs, stacks):
        """
        Call a layer of a api.
        """
        stmt = LayerStatement(
            layer, inputs, outputs, list(self._current_statement_ctxs), stacks
        )
        self.current_sir.add_statement(stmt)

    def call_AST(self, static_function, inputs, outputs, stacks):
        stmt = ASTStatement(
            static_function,
            inputs,
            outputs,
            list(self._current_statement_ctxs),
            stacks,
        )
        self.current_sir.add_statement(stmt)

    def get_sir(self, name: str):
        """
        Get a SIR from statement_factory.

        Args:
            name (str): the name of SIR.

        Returns:
            StatementIR: the SIR.
        """
        return self.statement_factory[name]

    @contextmanager
    def attach_statement_context_guard(self, ctx: StatementContext):
        """
        Attach a statement context to the current SIR.
        """
        self._current_statement_ctxs.append(ctx)
        try:
            yield
        finally:
            self._current_statement_ctxs.pop()

    def finalize(self, ret_vals):
        current_sir: StatementIR = self.current_sir
        current_sir.inputs = current_sir.analyse_inputs()
        current_sir.outputs = ret_vals
        log(2, "start subgraph compile and execution.\n")
        log(2, current_sir, "\n")
        return current_sir

    def compile_do_nothing(self) -> Callable[..., Any]:
        """
        Return a dummy function, which will return an empty list.

        Args:
            ret_vals (list[Symbol]): the return values of the function.
        """

        class DummyFunc:
            def __call__(*args, **kwargs):
                return []

            def graph_size(self):
                return 0

        return DummyFunc()

    def compile_fn(
        self, sir_name: str, input_spec: tuple[InputSpec | None, ...], **kwargs
    ):
        """
        start compile and return the python function, which must can be to_static without errors.
        """
        static_func = CompileSIRCache()(self, sir_name, input_spec, **kwargs)

        return static_func
