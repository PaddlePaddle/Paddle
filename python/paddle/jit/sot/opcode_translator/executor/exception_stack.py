# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
from typing import TYPE_CHECKING

from ...utils import InnerError
from .variables import ConstantVariable, ExceptionVariable

if TYPE_CHECKING:
    from .function_graph import FunctionGraph


@dataclasses.dataclass
class ExceptionStack:
    # This data structure manages exceptions as in CPython, primarily handling
    # the __context__ attribute of SotCapturedException.

    _exception_stack: list[ExceptionVariable] = dataclasses.field(
        default_factory=list
    )
    _current_exception: ExceptionVariable | None = dataclasses.field(
        default=None
    )

    def clear_current_exception(self):
        self._current_exception = None

    def set_current_exception(
        self, val: ExceptionVariable, graph: FunctionGraph
    ):
        self._set_context_and_break_context_reference_cycle(val, graph)
        self._current_exception = val

    def move_current_exception_to_stack(self):
        self.push(self.get_current_exception())
        self.clear_current_exception()

    def get_current_exception(self):
        if self._current_exception is None:
            raise InnerError("Current exception should not be None")
        return self._current_exception

    def _set_context_and_break_context_reference_cycle(
        self, val: ExceptionVariable, graph: FunctionGraph
    ):
        # set Exception.__context__
        self._set_context_recursive(val, len(self._exception_stack) - 1)
        self._break_context_reference_cycle(val, graph)

    def _set_context_recursive(self, val: ExceptionVariable, prev_idx: int):
        # Recursively sets the __context__ attribute for ExceptionVariable objects
        # in self._exception_stack. Ensures that __context__ is properly linked
        # to the previous exception in the stack.
        if (ctx := val.__context__) and not isinstance(ctx, ConstantVariable):
            return val
        if (
            len(self._exception_stack) + prev_idx > 0
        ):  # Prevent invalid negative indexing
            prev = self._exception_stack[prev_idx]
            self._set_context_recursive(prev, prev_idx - 1)
            val.setattr("__context__", prev)
        return val

    def _break_context_reference_cycle(
        self, val: ExceptionVariable, graph: FunctionGraph
    ):
        # Detects and breaks cycles in exception __context__ chains using Floyd's algorithm,
        # following CPython's implementation.

        fast = slow = val
        slow_update_toggle = False
        while True:
            context = fast.__context__
            if isinstance(
                context, ConstantVariable
            ):  # End of the chain; no context set
                break

            if context is val:
                # The chain loops back to the original exception; break the cycle.
                fast.setattr(
                    "__context__", ConstantVariable.wrap_literal(None, graph)
                )
                break

            fast = context
            if fast is slow:
                # Cycle detected; all exceptions on the path have been visited and checked.
                break

            if slow_update_toggle:
                slow = slow.__context__
            slow_update_toggle = not slow_update_toggle

    def pop(self) -> ExceptionVariable:
        return self._exception_stack.pop()

    def push(self, val: ExceptionVariable) -> None:
        self._exception_stack.append(val)

    def empty(self) -> bool:
        return len(self._exception_stack) == 0

    def __len__(self):
        return len(self._exception_stack)

    def __repr__(self):
        return f"ExceptionStack({self._exception_stack})"

    def __getitem__(self, idx: int) -> ExceptionVariable:
        return self._exception_stack[idx]

    def cleanup(self) -> None:
        self._exception_stack[:] = []
        self._current_exception = None
