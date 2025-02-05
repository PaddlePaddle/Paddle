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

from typing import TYPE_CHECKING, Any, Callable

from ..utils import log
from .compile_cache import CompileSIRCache
from .statement_ir import (
    ApiStatement,
    ASTStatement,
    CallStatement,
    LayerStatement,
    MethodStatement,
    StatementIR,
    StatementIRFactory,
    Symbol,
)

if TYPE_CHECKING:
    from paddle.static import InputSpec


class SymbolicTraceContext:
    """
    SymbolicTraceContext is a context manager, which is used to record the symbolic trace.

    """

    def __init__(self):
        self.reset()

    def reset(self):
        """
        Reset the context.
        """

        # TODO(dev): StatementIRFactory is a singleton, but SymbolicTraceContext is not.
        # whether will two different SymbolicTraceContext objects be conflict ?
        self.statement_factory = StatementIRFactory()
        self.sir_stack = [self.statement_factory.create()]

    @property
    def TOS(self):
        """
        The top SIR of sir_stack.

        Returns:
            StatementIR: the top of stack.
        """

        return self.sir_stack[-1]

    def call_SIR(self, sirname, inputs, outputs, stacks):
        """
        Call a SIR, which is a subgraph.
        """

        stmt = CallStatement(sirname, inputs, outputs, stacks)
        self.TOS.add_statement(stmt)

    def call_API(self, api, inputs, outputs, stacks):
        """
        Call a paddle api.
        """
        assert callable(api), "call_API must receive a paddle api."
        stmt = ApiStatement(api, inputs, outputs, stacks)
        self.TOS.add_statement(stmt)

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
        stmt = MethodStatement(method_name, inputs, outputs, stacks)
        self.TOS.add_statement(stmt)

    def call_LAYER(self, layer, inputs, outputs, stacks):
        """
        Call a layer of a api.
        """
        stmt = LayerStatement(layer, inputs, outputs, stacks)
        self.TOS.add_statement(stmt)

    def call_AST(self, static_function, inputs, outputs, stacks):
        stmt = ASTStatement(static_function, inputs, outputs, stacks)
        self.TOS.add_statement(stmt)

    def get_sir(self, name: str):
        """
        Get a SIR from statement_factory.

        Args:
            name (str): the name of SIR.

        Returns:
            StatementIR: the SIR.
        """
        return self.statement_factory[name]

    def reset_TOS(self):
        """
        Reset the TOS.
        """
        self.sir_stack.pop()
        self.sir_stack.append(self.statement_factory.create())

    def replace_TOS(self, sir):
        """
        Use deepcopyed sir to replace the TOS.
        This function will update statement_factory.
        """
        self.sir_stack.pop()
        self.sir_stack.append(sir)
        self.statement_factory.update(sir)

    def return_TOS(self, ret_vals):
        cur_sir: StatementIR = self.TOS
        cur_sir.inputs = cur_sir.analyse_inputs()
        cur_sir.outputs = ret_vals
        log(2, "start subgraph compile and execution.\n")
        log(2, self.TOS, "\n")
        return cur_sir

    def compile_do_nothing(self) -> Callable[[...], Any]:
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
        self, sir_name: str, input_spec: tuple[InputSpec, ...], **kwargs
    ):
        """
        start compile and return the python function, which must can be to_static without errors.
        """
        static_func = CompileSIRCache()(self, sir_name, input_spec, **kwargs)

        return static_func
