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

import inspect
import sys
from typing import TYPE_CHECKING

import paddle

from ...utils import (
    BreakGraphError,
    DataDependencyControlFlowBreak,
    FallbackError,
    UnsupportedIteratorBreak,
)
from ..instruction_utils import Instruction
from .dispatch_functions import generator_send
from .guard import StringifiedExpression, union_free_vars
from .opcode_executor import OpcodeExecutorBase, Stop
from .tracker import DanglingTracker, Tracker
from .variables import (
    BuiltinVariable,
    ConstantVariable,
    GeneratorVariable,
    IterVariable,
    ObjectVariable,
    UserDefinedIterVariable,
    VariableBase,
)

if TYPE_CHECKING:
    from .function_graph import FunctionGraph
    from .pycode_generator import PyCodeGen
    from .variables import FunctionVariable
    from .virtual_frame import VirtualFrame


class FunctionGlobalTracker(Tracker):
    """
    A tracker class that represents a function global variable.

    Args:
        fn: FunctionVariable object.
        name: The name of the global variable.

    """

    def __init__(self, fn: FunctionVariable, name: str):
        super().__init__([fn])
        self.fn = fn
        self.name = name

    def gen_instructions(self, codegen: PyCodeGen):
        """
        Generate bytecode instructions in order to put the variables at the top of the stack.

        Args:
            codegen: The PyCodeGen object used to generate bytecode.

        """
        self.fn.tracker.gen_instructions(codegen)
        codegen.gen_load_attr("__globals__")
        codegen.gen_load_const(self.name)
        codegen.gen_subscribe()

    def guard_tree_expr_node(self) -> paddle.framework.core.ExprNodeBase:
        fn_tracer = self.fn.tracker.guard_tree_expr_node()
        return paddle.framework.core.ItemExprNode(
            paddle.framework.core.AttributeExprNode(
                fn_tracer,
                "__globals__",
            ),
            paddle.framework.core.ConstantExprNode(self.name),
        )

    def trace_value_from_frame(self) -> StringifiedExpression:
        """
        Trace the value of the function global variable from the frame.

        Returns:
            StringifiedExpression: The traced value of the function global variable.

        """
        fn_tracer = self.fn.tracker.trace_value_from_frame()
        return StringifiedExpression(
            f"{{}}.__globals__['{self.name}']",
            [fn_tracer],
            union_free_vars(fn_tracer.free_vars),
        )

    def __repr__(self) -> str:
        return f"FunctionGlobalTracker(fn={self.fn}, name={self.name})"


def inline_for_iter_impl(exe: OpcodeExecutorBase, instr: Instruction):
    iterator = exe.stack.top
    assert isinstance(iterator, IterVariable)

    exe._graph.add_global_guarded_variable(iterator)

    # simply get next
    if not isinstance(iterator, UserDefinedIterVariable):
        try:
            exe.stack.push(iterator.next())
        except StopIteration:
            exe.stack.pop()
            assert isinstance(instr.jump_to, Instruction)
            exe.vframe.lasti = exe.indexof(instr.jump_to)
            if sys.version_info >= (3, 12):
                assert exe._instructions[exe.vframe.lasti].opname == "END_FOR"
                skip_n_instrs = 2 if sys.version_info >= (3, 13) else 1
                exe.vframe.lasti += skip_n_instrs

    else:
        exe._graph.remove_global_guarded_variable(iterator)
        raise BreakGraphError(
            UnsupportedIteratorBreak(
                reason_str=f"Found {iterator.__class__.__name__} as iterator."
            )
        )


class OpcodeInlineExecutor(OpcodeExecutorBase):
    """
    A class that represents an executor for inlined opcode operations.

    Args:
        fn_variable: The function variable.

    """

    def __init__(
        self,
        vframe: VirtualFrame,
        code_var: VariableBase,
        graph: FunctionGraph,
    ):
        super().__init__(vframe, graph)
        self.return_value: VariableBase | None = None
        self._code_var = code_var
        self._name = "InlineFn"

    def inline_call(self) -> VariableBase:
        """
        Execute the inline call of the function.
        """
        self._graph.add_global_guarded_variable(self._code_var)
        self.run()
        assert self.return_value is not None
        return self.return_value

    def RETURN_VALUE(self, instr: Instruction):
        assert (
            len(self.stack) == 1
        ), f"Stack must have one element, but get {len(self.stack)} elements."
        self.return_value = self.stack.pop()
        return Stop(state="Return")

    def RETURN_CONST(self, instr: Instruction):
        self.return_value = self.vframe.consts[instr.arg]
        return Stop(state="Return")

    def _break_graph_when_if(self, result, instr: Instruction):
        """
        Helper method to raise a BreakGraphError when breaking the graph in a jump operation.

        Args:
            result: The result of the operation.
            instr (Instruction): The jump instruction.
        """

        raise BreakGraphError(DataDependencyControlFlowBreak())

    def FOR_ITER(self, instr: Instruction):
        return inline_for_iter_impl(self, instr)


class OpcodeInlineGeneratorExecutor(OpcodeExecutorBase):
    def __init__(
        self,
        vframe: VirtualFrame,
        code_var: VariableBase,
        graph: FunctionGraph,
    ):
        super().__init__(vframe, graph)
        self.return_value: VariableBase | None = None
        self._code_var = code_var
        self._name = "InlineGen"

    def inline_call(self) -> VariableBase:
        self._graph.add_global_guarded_variable(self._code_var)
        self.run()
        assert self.return_value is not None
        return self.return_value

    def RETURN_GENERATOR(self, instr: Instruction):
        vframe = self.vframe
        code_var = self._code_var
        # NOTE: we set the real tracker in calling function
        self.return_value = GeneratorVariable(
            code_var, vframe, self._graph, DanglingTracker()
        )
        return Stop(state="Return")

    def SEND(self, instr: Instruction):
        assert len(self.stack) >= 2
        recv = self.stack.pop()
        source_obj = self.stack.top
        if not isinstance(source_obj, IterVariable):
            raise FallbackError(
                "Yield from for non-generator object is not supported."
            )
        self.stack.push(
            BuiltinVariable(generator_send, self._graph, DanglingTracker())(
                source_obj, recv
            )
        )

    def END_SEND(self, instr: Instruction):
        value = self.stack.pop()
        receiver = self.stack.pop()  # pop the receiver
        self.stack.push(value)

    def GEN_START(self, instr: Instruction):
        tos = self.stack.pop()
        assert isinstance(tos, ConstantVariable)
        assert tos.value is None

    def YIELD_VALUE(self, instr: Instruction):
        assert len(self.stack) >= 1
        self.return_value = self.stack.pop()
        return Stop(state="Yield")

    def GET_YIELD_FROM_ITER(self, instr: Instruction):
        source_obj = self.stack.top
        if isinstance(source_obj, ObjectVariable) and inspect.iscoroutine(
            source_obj.value
        ):
            raise FallbackError(
                "Get yield from iter for coroutine object is not supported."
            )
        if isinstance(source_obj, GeneratorVariable):
            return
        source_obj = self.stack.pop()
        iter_variable = BuiltinVariable(iter, self._graph, DanglingTracker())(
            source_obj
        )
        self.stack.push(iter_variable)

    def YIELD_FROM(self, instr: Instruction):
        recv = self.stack.pop()
        source_obj = self.stack.top
        if not isinstance(source_obj, IterVariable):
            raise FallbackError(
                "Yield from for non-generator object is not supported."
            )
        self.return_value = BuiltinVariable(
            generator_send, self._graph, DanglingTracker()
        )(source_obj, recv)
        assert self.vframe.lasti > 0
        self.vframe.lasti -= 1
        return Stop(state="Yield")

    def FOR_ITER(self, instr: Instruction):
        return inline_for_iter_impl(self, instr)

    def RETURN_VALUE(self, instr: Instruction):
        assert (
            len(self.stack) == 1
        ), f"Stack must have one element, but get {len(self.stack)} elements."
        self.return_value = self.stack.pop()
        return Stop(state="Return")

    def RETURN_CONST(self, instr: Instruction):
        self.return_value = self.vframe.consts[instr.arg]
        return Stop(state="Return")
