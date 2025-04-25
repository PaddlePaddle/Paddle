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

import builtins
import inspect
import re
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, NamedTuple

import paddle

from ...utils import log
from .guard import StringifiedExpression, union_free_vars
from .tracker import (
    BuiltinTracker,
    CellTracker,
    ConstTracker,
    DanglingTracker,
    DummyTracker,
    LocalTracker,
    Tracker,
)
from .variable_stack import VariableStack
from .variables.base import VariableBase, VariableFactory
from .variables.basic import (
    CellVariable,
    FunctionGlobalVariable,
    GlobalVariable,
    NullVariable,
)

if TYPE_CHECKING:
    import types

    from typing_extensions import TypeAlias

    from .function_graph import FunctionGraph
    from .pycode_generator import PyCodeGen
    from .variables.callable import FunctionVariable

    # The type to represent the (*args, **kwargs) pack in the call.
    CallArgsPack: TypeAlias = tuple[tuple[Any, ...], dict[str, Any]]


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

    def guard_tree_expr_node(self) -> paddle.framework.core.ExprNodeBase:
        fn_tracer = self.fn.tracker.guard_tree_expr_node()
        return paddle.framework.core.AttributeExprNode(
            paddle.framework.core.ItemExprNode(
                paddle.framework.core.AttributeExprNode(
                    fn_tracer,
                    "__closure__",
                ),
                paddle.framework.core.ConstantExprNode(self.idx),
            ),
            "cell_contents",
        )

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


@contextmanager
def signature_clear_guard(fn, name):
    if not hasattr(fn, name):
        yield
    else:
        saved_attr = getattr(fn, name)
        delattr(fn, name)
        yield
        setattr(fn, name, saved_attr)


def validate_value(value):
    assert isinstance(
        value, VariableBase
    ), f"value: {value}, type should be VariableBase(or derived), but get {type(value)}"
    assert not isinstance(value.tracker, DanglingTracker) or isinstance(
        value, (NullVariable, CellVariable)
    ), f"dangling variable {value} should not be pushed into stack."


class VirtualFrameState(NamedTuple):
    locals: dict[str, VariableBase]
    builtins: dict[str, VariableBase]
    cells: dict[str, VariableBase]
    lasti: int
    stack_data: list[VariableBase]


class VirtualFrame:
    code: types.CodeType
    locals: dict[str, Any]  # TODO: should we use DictVariable instead of dict?
    globals: GlobalVariable
    builtins: dict[str, Any]
    consts: list[Any]
    cells: dict[str, Any]
    lasti: int
    stack: VariableStack

    def __init__(self, code: types.CodeType):
        self.code = code
        self.locals = {}
        self.globals = None  # type: ignore
        self.builtins = {}
        self.cells = {}
        self.lasti = 0
        self.consts = []
        self.stack = VariableStack(validate_value_func=validate_value)

    @staticmethod
    def from_real_frame(frame: types.FrameType, graph: FunctionGraph):
        code = frame.f_code
        locals = frame.f_locals
        vframe = VirtualFrame(code)

        # convert locals
        free_or_cell_vars = code.co_cellvars + code.co_freevars
        for name, value in locals.items():
            tracker = (
                CellTracker(name)
                if name in free_or_cell_vars
                else LocalTracker(name)
            )
            vframe.locals[name] = VariableFactory.from_value(
                value, graph, tracker
            )

        for name in free_or_cell_vars:
            # create a cell for each variable.
            vframe.cells[name] = CellVariable()  # put in cells.
            if name in vframe.locals:
                vframe.cells[name].set_value(vframe.locals[name])

        # convert globals
        vframe.globals = GlobalVariable(
            frame.f_globals,
            graph,
            DanglingTracker(),
        )

        # convert builtins
        for name, value in builtins.__dict__.items():
            vframe.builtins[name] = VariableFactory.from_value(
                value, graph, BuiltinTracker(name)
            )
        # Temporarily use the builtins from the graph to avoid the conversion overhead.
        graph.builtins = vframe.builtins

        # prepare consts
        for value in code.co_consts:
            vframe.consts.append(
                VariableFactory.from_value(value, graph, ConstTracker(value))
            )
        return vframe

    @staticmethod
    def from_inline_call(
        code: types.CodeType,
        fn_var: FunctionVariable,
        fn_value: types.FunctionType,
        graph: FunctionGraph,
        call_args_pack: CallArgsPack,
    ):
        call_args, call_kwargs = call_args_pack
        vframe = VirtualFrame(code)
        vframe.globals = FunctionGlobalVariable(
            fn_var,
            fn_value.__globals__,
            graph,
            DanglingTracker(),
        )

        # convert builtins
        # NOTE(SigureMo): inline call should inherit the builtins from the caller to reduce the conversion overhead.
        vframe.builtins = graph.builtins

        # prepare consts
        for value in code.co_consts:
            vframe.consts.append(
                VariableFactory.from_value(value, graph, ConstTracker(value))
            )

        # convert locals
        # temparay clear the fn.__signature__ to avoid signature check error
        with signature_clear_guard(
            fn_value, "__signature__"
        ), signature_clear_guard(fn_value, "__wrapped__"):
            sig = inspect.signature(fn_value)
            bound_args = sig.bind(*call_args, **call_kwargs)
        bound_args.apply_defaults()
        for name, value in bound_args.arguments.items():
            assert name in sig.parameters
            # Convert varargs and kwargs to Variable
            if sig.parameters[name].kind == inspect.Parameter.VAR_POSITIONAL:
                tracker = DummyTracker(value)
            elif sig.parameters[name].kind == inspect.Parameter.VAR_KEYWORD:
                tracker = DummyTracker(list(value.values()))
            # Convert default args to Variable
            elif not isinstance(value, VariableBase):
                tracker = ConstTracker(value)
            else:
                tracker = value.tracker
            value = VariableFactory.from_value(value, graph, tracker)
            vframe.locals[name] = value

        log(
            5,
            f"[INLINE CALL] {code.co_name} with locals: ",
            vframe.locals,
        )

        # handle implicit variables in comprehensions
        vframe.handle_comps(fn_value)

        # convert closures
        closure = fn_var.get_py_value().__closure__
        for name in code.co_cellvars + code.co_freevars:
            # create a cell for each variable.
            vframe.cells[name] = CellVariable()  # put in cells.
            if name in vframe.locals:
                vframe.cells[name].set_value(vframe.locals[name])

        if closure is None:
            return vframe
        assert len(closure) == len(code.co_freevars)
        for idx, (name, cell) in enumerate(zip(code.co_freevars, closure)):
            value = cell.cell_contents
            value = VariableFactory.from_value(
                value, graph, FunctionClosureTracker(fn_var, idx)
            )
            # wrapped by a CellVariable
            if not isinstance(value, CellVariable):
                value = CellVariable(value)
            vframe.cells[name] = value
        return vframe

    def handle_comps(self, fn_value):
        is_comp = any(
            x in fn_value.__name__
            for x in ['<listcomp>', '<dictcomp>', '<setcomp>', '<genexpr>']
        )
        if not is_comp:
            return
        pattern = r'implicit\d+'
        for name in list(self.locals.keys()):
            if re.match(pattern, name):
                self.locals[name.replace('implicit', '.')] = self.locals[name]

    def get_state(self):
        return VirtualFrameState(
            locals=self.locals.copy(),
            builtins=self.builtins.copy(),
            cells=self.cells.copy(),
            lasti=self.lasti,
            stack_data=list(self.stack._data),
        )

    def restore_state(self, state: VirtualFrameState):
        self.locals = state.locals
        self.builtins = state.builtins
        self.cells = state.cells
        self.lasti = state.lasti
        self.stack._data = state.stack_data
