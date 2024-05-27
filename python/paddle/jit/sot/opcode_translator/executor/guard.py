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

import types
import weakref
from typing import TYPE_CHECKING, Any, Callable, TypeVar

import paddle

from ...profiler import EventGuard
from ...utils import current_tmp_name_records, log, log_do

Guard = Callable[[types.FrameType], bool]

if TYPE_CHECKING:
    from .variables import VariableBase

    CheckGuardInputT = TypeVar("CheckGuardInputT", bound=VariableBase)

# NOTE(SigureMo): [How to write Stringify Guard?]
# 1. we should capture free variables manually, the string cannot capture free
#    variables automatically.
# 2. Be aware that the comparison logic before and after stringify may be different.
# 3. we should compute as much as possible at "compile time" and encode the
#    computation in the Guard string, rather than passing it to runtime to minimize
#    runtime overhead.


class StringifyExpression:
    """
    Used to store string based expressions for generating Guard.
    """

    def __init__(self, str_expr, sub_exprs, free_vars):
        expr = str_expr.format(*[arg.expr for arg in sub_exprs])
        self.expr = current_tmp_name_records().add_tmp_var(expr)
        self.inlined_expr = str_expr.format(
            *[arg.inlined_expr for arg in sub_exprs]
        )
        self.free_vars = free_vars

    def __hash__(self):
        if self.free_vars:
            return hash((self.inlined_expr, id(self)))
        else:
            return hash(self.inlined_expr)


def union_free_vars(*free_vars: dict[str, Any]):
    return {k: v for d in free_vars for k, v in d.items()}


def make_guard(stringify_guards: list[StringifyExpression]) -> Guard:
    """
    Make a guard from a list of StringifyExpression.

    For more design ideas, refer to the `Stringify guard <https://github.com/PaddlePaddle/PaddleSOT/blob/develop/docs/design/stringify-guard.md>`_ for details.

    Args:
        stringify_guards: a list of StringifyExpression.
    """
    with EventGuard("make_guard"):
        num_guards = len(stringify_guards)
        if not num_guards:
            guard = lambda frame: True
            guard.expr = "lambda frame: True"
            return guard

        def analyse_expressions(stringify_exprs, tmp_names):
            func_string = "def built_guard_fn(frame):\n"
            lambda_string = "lambda frame: "
            free_vars = {}

            for k, v in tmp_names.items():
                func_string += f"    {v} = {k}\n"

            func_result = ""
            for str_expr in stringify_exprs:
                func_result += str_expr.expr + " and "
                lambda_string += str_expr.inlined_expr + " and "
                free_vars = union_free_vars(free_vars, str_expr.free_vars)

            func_string += f"    return {func_result[:-5]}"

            return func_string, free_vars, lambda_string[:-5]

        (
            func_string,
            free_vars,
            lambda_string,
        ) = analyse_expressions(
            stringify_guards, current_tmp_name_records().tmp_names_record
        )

        exec(
            func_string,
            free_vars,
        )

        guard = free_vars['built_guard_fn']
        log(3, f"[Guard]: {lambda_string}\n")
        guard.lambda_expr = lambda_string
        guard.expr = func_string
        assert callable(guard), "guard must be callable."

        return guard


def support_weak_ref(obj):
    if isinstance(obj, types.FunctionType):
        return True
    return False


def check_guard(
    fn: Callable[[CheckGuardInputT], list[StringifyExpression]]
) -> Callable[[CheckGuardInputT], list[StringifyExpression]]:
    def wrapper(self: CheckGuardInputT) -> list[StringifyExpression]:
        assert (
            self.tracker.is_traceable()
        ), "Cannot make guard from a non-tracable guard variable."

        def guard_log():
            frame_value_tracer = self.tracker.trace_value_from_frame()
            print(
                f"[Guard]: guard_fn for {self}, tracker={self.tracker.__class__.__name__}, value={frame_value_tracer.expr}"
            )

        log_do(4, guard_log)
        return fn(self)

    return wrapper


@check_guard
def object_equal_stringify_guard(self) -> list[StringifyExpression]:
    frame_value_tracer = self.tracker.trace_value_from_frame()

    obj_free_var_name = f"__{self.id}"
    weak_ref_obj = self.get_py_value()
    if support_weak_ref(weak_ref_obj):
        weak_ref_obj = weakref.ref(self.get_py_value())
        return [
            StringifyExpression(
                f"{obj_free_var_name}() is not None and {{}} == {obj_free_var_name}()",
                [frame_value_tracer],
                union_free_vars(
                    frame_value_tracer.free_vars,
                    {obj_free_var_name: weak_ref_obj},
                ),
            )
        ]
    return [
        StringifyExpression(
            f"{{}} == {obj_free_var_name}",
            [frame_value_tracer],
            union_free_vars(
                frame_value_tracer.free_vars,
                {obj_free_var_name: self.get_py_value()},
            ),
        )
    ]


def stringify_pyobject(obj: object) -> tuple[str, dict[str, Any]]:
    if isinstance(obj, paddle.core.VarDesc.VarType):
        return f"paddle.core.VarDesc.VarType({obj.value})", {"paddle": paddle}
    elif isinstance(obj, paddle.core.DataType):
        return f"paddle.core.DataType({obj.value})", {"paddle": paddle}
    # For builtin values
    return f"{obj!r}", {}
