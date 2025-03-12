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

import sys
from typing import TYPE_CHECKING

from ...utils import (
    BreakGraphError,
    DataDependencyControlFlowBreak,
    UnsupportedIteratorBreak,
)
from ..instruction_utils import Instruction
from .guard import StringifiedExpression, union_free_vars
from .opcode_executor import OpcodeExecutorBase, Stop
from .tracker import Tracker
from .variables import (
    IterVariable,
    SequenceIterVariable,
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


class FunctionClosureTracker(Tracker):
    """
    A tracker class that represents a function closure variable.

    Args:
        fn: The FunctionVariable object.
        idx: The index of the closure variable.

    """

    def __init__(self, fn: FunctionVariable, idx: int):
        super().__init__([fn])
        self.fn = fn
        self.idx = idx

    def gen_instructions(self, codegen: PyCodeGen):
        """
        Generate bytecode instructions to trace the value of the function closure variable.

        Args:
            codegen: The PyCodeGen object used to generate bytecode.

        """
        self.fn.tracker.gen_instructions(codegen)
        codegen.gen_load_attr("__closure__")
        codegen.gen_load_const(self.idx)
        codegen.gen_subscribe()
        codegen.gen_load_attr("cell_contents")

    def trace_value_from_frame(self):
        """
        Trace the value of the function closure variable from the frame.

        Returns:
            The traced value of the function closure variable.

        """
        fn_tracer = self.fn.tracker.trace_value_from_frame()
        return StringifiedExpression(
            f"{{}}.__closure__[{self.idx}].cell_contents",
            [fn_tracer],
            union_free_vars(fn_tracer.free_vars),
        )

    def __repr__(self) -> str:
        return f"FunctionClosureTracker(fn={self.fn}, idx={self.idx})"


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
        self._name = "Inline"

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
        iterator = self.stack.top
        assert isinstance(iterator, IterVariable)

        self._graph.add_global_guarded_variable(iterator)

        # simply get next
        if isinstance(
            iterator,
            SequenceIterVariable,
        ):
            try:
                self.stack.push(iterator.next())
            except StopIteration:
                self.stack.pop()
                assert isinstance(instr.jump_to, Instruction)
                self.vframe.lasti = self.indexof(instr.jump_to)
                if sys.version_info >= (3, 12):
                    assert (
                        self._instructions[self.vframe.lasti].opname
                        == "END_FOR"
                    )
                    skip_n_instrs = 2 if sys.version_info >= (3, 13) else 1
                    self.vframe.lasti += skip_n_instrs

        else:
            self._graph.remove_global_guarded_variable(iterator)
            raise BreakGraphError(
                UnsupportedIteratorBreak(
                    reason_str=f"Found {iterator.__class__.__name__} as iterator."
                )
            )
