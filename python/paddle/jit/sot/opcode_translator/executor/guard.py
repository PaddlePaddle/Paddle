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
from dataclasses import dataclass
from enum import IntEnum
from functools import cached_property
from typing import TYPE_CHECKING, Any, Callable, TypeVar

import paddle

from ...profiler import EventGuard
from ...utils import (
    ENV_SOT_ENABLE_COMPILED_GUARD,
    ENV_SOT_ENABLE_STRICT_GUARD_CHECK,
    current_symbol_registry,
    log,
    log_do,
)

Guard = Callable[[types.FrameType], bool]


class GuardAccessKind(IntEnum):
    LOCAL = 0
    GLOBAL = 1
    BUILTIN = 2
    CONSTANT = 3
    ATTR = 4
    ITEM = 5


class GuardAttrKind(IntEnum):
    GENERIC = 0
    TRAINING = 1
    SUB_LAYERS = 2
    FORWARD_PRE_HOOKS = 3
    FORWARD_POST_HOOKS = 4
    FUNC = 5
    CODE = 6
    GLOBALS = 7
    CALL = 8
    FORWARD = 9
    STOP_GRADIENT = 10


class GuardOpKind(IntEnum):
    GRAD_ENABLED = 0
    TYPE_MATCH = 1
    INSTANCE_CHECK = 2
    ID_MATCH = 3
    VALUE_MATCH = 4
    LENGTH_MATCH = 5
    LAYER_MATCH = 6
    LAYER_MATCH_GROUP = 7
    TENSOR_SHAPE = 8
    TENSOR_DTYPE = 9
    TENSOR_IS_DIST = 10
    TENSOR_META = 11
    TENSOR_DIST_META = 12
    TENSOR_NOT_HOLD_ALLOCATION = 13
    NUMPY_DTYPE = 14
    NUMPY_SHAPE = 15
    WEAKREF_MATCH = 16
    EXPR_MATCH = 17


class GuardExprKind(IntEnum):
    CONSTANT = 0
    ACCESS = 1
    UNARY = 2
    BINARY = 3


class GuardUnaryOp(IntEnum):
    POSITIVE = 0
    NEGATIVE = 1
    BITWISE_NOT = 2
    LOGICAL_NOT = 3
    BOOL = 4

    @classmethod
    def from_symbol(cls, symbol: str) -> GuardUnaryOp:
        return {
            "+": cls.POSITIVE,
            "-": cls.NEGATIVE,
            "~": cls.BITWISE_NOT,
            "not": cls.LOGICAL_NOT,
            "bool": cls.BOOL,
        }[symbol]


class GuardBinaryOp(IntEnum):
    EQ = 0
    NE = 1
    LT = 2
    LE = 3
    GT = 4
    GE = 5
    ADD = 6
    SUB = 7
    MUL = 8
    TRUE_DIV = 9
    FLOOR_DIV = 10
    MOD = 11
    POW = 12
    LSHIFT = 13
    RSHIFT = 14
    BITWISE_AND = 15
    BITWISE_OR = 16
    BITWISE_XOR = 17

    @classmethod
    def from_symbol(cls, symbol: str) -> GuardBinaryOp:
        return {
            "==": cls.EQ,
            "!=": cls.NE,
            "<": cls.LT,
            "<=": cls.LE,
            ">": cls.GT,
            ">=": cls.GE,
            "+": cls.ADD,
            "-": cls.SUB,
            "*": cls.MUL,
            "/": cls.TRUE_DIV,
            "//": cls.FLOOR_DIV,
            "%": cls.MOD,
            "**": cls.POW,
            "<<": cls.LSHIFT,
            ">>": cls.RSHIFT,
            "&": cls.BITWISE_AND,
            "|": cls.BITWISE_OR,
            "^": cls.BITWISE_XOR,
        }[symbol]


_SPECIAL_ATTR_KINDS = {
    "training": GuardAttrKind.TRAINING,
    "_sub_layers": GuardAttrKind.SUB_LAYERS,
    "_forward_pre_hooks": GuardAttrKind.FORWARD_PRE_HOOKS,
    "_forward_post_hooks": GuardAttrKind.FORWARD_POST_HOOKS,
    "__func__": GuardAttrKind.FUNC,
    "__code__": GuardAttrKind.CODE,
    "__globals__": GuardAttrKind.GLOBALS,
    "__call__": GuardAttrKind.CALL,
    "forward": GuardAttrKind.FORWARD,
    "stop_gradient": GuardAttrKind.STOP_GRADIENT,
}


@dataclass(frozen=True)
class GuardAccessStep:
    kind: GuardAccessKind
    value: Any
    attr_kind: GuardAttrKind = GuardAttrKind.GENERIC

    @staticmethod
    def local(name: str) -> GuardAccessStep:
        return GuardAccessStep(GuardAccessKind.LOCAL, name)

    @staticmethod
    def global_(name: str) -> GuardAccessStep:
        return GuardAccessStep(GuardAccessKind.GLOBAL, name)

    @staticmethod
    def builtin(name: str) -> GuardAccessStep:
        return GuardAccessStep(GuardAccessKind.BUILTIN, name)

    @staticmethod
    def constant(value: Any) -> GuardAccessStep:
        return GuardAccessStep(GuardAccessKind.CONSTANT, value)

    @staticmethod
    def attr(name: str) -> GuardAccessStep:
        return GuardAccessStep(
            GuardAccessKind.ATTR,
            name,
            _SPECIAL_ATTR_KINDS.get(name, GuardAttrKind.GENERIC),
        )

    @staticmethod
    def item(key: Any) -> GuardAccessStep:
        return GuardAccessStep(GuardAccessKind.ITEM, key)

    def to_cpp_step(self) -> tuple[Any, ...]:
        if self.kind is GuardAccessKind.ATTR:
            return (int(self.kind), self.value, int(self.attr_kind))
        if self.attr_kind is not GuardAttrKind.GENERIC:
            raise UnsupportedCompiledGuard(
                f"{self.kind.name} access cannot carry attr kind {self.attr_kind.name}"
            )
        return (int(self.kind), self.value)


GuardAccess = tuple[GuardAccessStep, ...]


@dataclass(frozen=True)
class GuardExpr:
    kind: GuardExprKind
    value: Any = None
    access: GuardAccess | None = None
    unary_op: GuardUnaryOp | None = None
    binary_op: GuardBinaryOp | None = None
    lhs: GuardExpr | None = None
    rhs: GuardExpr | None = None

    @staticmethod
    def constant(value: Any) -> GuardExpr:
        return GuardExpr(GuardExprKind.CONSTANT, value=value)

    @staticmethod
    def from_access(access: GuardAccess) -> GuardExpr:
        return GuardExpr(GuardExprKind.ACCESS, access=access)

    @staticmethod
    def unary(op: GuardUnaryOp, operand: GuardExpr) -> GuardExpr:
        return GuardExpr(GuardExprKind.UNARY, unary_op=op, lhs=operand)

    @staticmethod
    def binary(op: GuardBinaryOp, lhs: GuardExpr, rhs: GuardExpr) -> GuardExpr:
        return GuardExpr(GuardExprKind.BINARY, binary_op=op, lhs=lhs, rhs=rhs)

    def to_cpp_expr(self) -> tuple[Any, ...]:
        if self.kind is GuardExprKind.CONSTANT:
            return (int(self.kind), self.value)
        if self.kind is GuardExprKind.ACCESS:
            assert self.access is not None
            return (
                int(self.kind),
                tuple(step.to_cpp_step() for step in self.access),
            )
        if self.kind is GuardExprKind.UNARY:
            assert self.unary_op is not None
            assert self.lhs is not None
            return (
                int(self.kind),
                int(self.unary_op),
                self.lhs.to_cpp_expr(),
            )
        if self.kind is GuardExprKind.BINARY:
            assert self.binary_op is not None
            assert self.lhs is not None
            assert self.rhs is not None
            return (
                int(self.kind),
                int(self.binary_op),
                self.lhs.to_cpp_expr(),
                self.rhs.to_cpp_expr(),
            )
        raise UnsupportedCompiledGuard(f"unknown guard expr kind: {self.kind}")


@dataclass(frozen=True)
class GuardSpec:
    kind: GuardOpKind
    access: GuardAccess | None = None
    args: tuple[Any, ...] = ()

    @staticmethod
    def grad_enabled(value: bool) -> GuardSpec:
        return GuardSpec(GuardOpKind.GRAD_ENABLED, args=(value,))

    @staticmethod
    def expr_match(expr: GuardExpr) -> GuardSpec:
        return GuardSpec(GuardOpKind.EXPR_MATCH, args=(expr,))

    def to_cpp_spec(self) -> tuple[Any, ...]:
        if self.kind is GuardOpKind.GRAD_ENABLED:
            return (int(self.kind), *self.args)
        if self.kind is GuardOpKind.EXPR_MATCH:
            expr = self.args[0]
            assert isinstance(expr, GuardExpr)
            return (int(self.kind), expr.to_cpp_expr())
        if not self.access:
            raise UnsupportedCompiledGuard(
                f"{self.kind.name} guard requires an access path"
            )
        return (
            int(self.kind),
            tuple(step.to_cpp_step() for step in self.access),
            *self.args,
        )


class UnsupportedCompiledGuard(ValueError):
    pass


if TYPE_CHECKING:
    from .variables import VariableBase

    CheckGuardInputT = TypeVar("CheckGuardInputT", bound=VariableBase)

# NOTE(SigureMo): [How to write Stringified Guard?]
# 1. we should capture free variables manually, the string cannot capture free
#    variables automatically.
# 2. Be aware that the comparison logic before and after stringify may be different.
# 3. we should compute as much as possible at "compile time" and encode the
#    computation in the Guard string, rather than passing it to runtime to minimize
#    runtime overhead.


class StringifiedExpression:
    """
    Used to store string based expressions for generating Guard.
    """

    def __init__(
        self,
        expr_template: str,
        sub_exprs: list[StringifiedExpression],
        free_vars: dict[str, Any],
    ):
        self.expr_template = expr_template
        expr = self.expr_template.format(
            *[sub_expr.symbol for sub_expr in sub_exprs]
        )
        self.registered_expr = expr
        self.symbol = current_symbol_registry().request_symbol(expr)
        self.sub_exprs = sub_exprs
        self.free_vars = free_vars

    @cached_property
    def inlined_expr(self):
        return self.expr_template.format(
            *[sub_expr.inlined_expr for sub_expr in self.sub_exprs]
        )

    def gen_expr(self):
        def gen_expr_fn():
            return self.expr_template.format(
                *[sub_expr.gen_expr() for sub_expr in self.sub_exprs]
            )

        return current_symbol_registry().gen_expr(
            self.registered_expr, gen_expr_fn
        )

    def __hash__(self):
        if self.free_vars:
            return hash((self.inlined_expr, id(self)))
        else:
            return hash(self.inlined_expr)


def union_free_vars(*free_vars: dict[str, Any]):
    return {k: v for d in free_vars for k, v in d.items()}


def make_guard(stringified_guards: list[StringifiedExpression]) -> Guard:
    """
    Make a guard from a list of StringifiedExpression.

    For more design ideas, refer to the `Stringified guard <https://github.com/PaddlePaddle/PaddleSOT/blob/develop/docs/design/stringify-guard.md>`_ for details.

    Args:
        stringified_guards: a list of StringifiedExpression.
    """
    with EventGuard("make_guard"):
        num_guards = len(stringified_guards)
        if not num_guards:
            guard = lambda frame: True
            guard.expr = "lambda frame: True"
            guard.original_guard = guard
            if ENV_SOT_ENABLE_STRICT_GUARD_CHECK.get():
                guard.mirror_guard = lambda frame: True
            return guard

        free_vars = union_free_vars(
            *(expr.free_vars for expr in stringified_guards)
        )
        inlined_guard_expr = "lambda frame: " + " and ".join(
            [expr.inlined_expr for expr in stringified_guards]
        )
        guard_expr: str = "lambda frame: " + " and ".join(
            [expr.gen_expr() for expr in stringified_guards]
        )

        guard = eval(guard_expr, free_vars)

        log(3, f"[Guard] {inlined_guard_expr}\n")
        guard.inlined_expr = inlined_guard_expr
        guard.expr = guard_expr

        def check_guard_callable(guard: Guard):
            assert callable(guard), "guard must be callable."

        if ENV_SOT_ENABLE_STRICT_GUARD_CHECK.get():
            mirror_guard_expr_list: list[str] = []
            mirror_guard_temp_free_vars: dict[str, Any] = {}
            for expr in stringified_guards:
                mirror_guard_expr_list.append(expr.inlined_expr)
                mirror_guard_temp_free_vars.update(expr.free_vars)
            mirror_guard_expr = "lambda frame: " + " and ".join(
                mirror_guard_expr_list
            )
            mirror_guard_free_vars = union_free_vars(
                mirror_guard_temp_free_vars
            )
            guard.mirror_guard = eval(mirror_guard_expr, mirror_guard_free_vars)
            guard.mirror_guard.expr = mirror_guard_expr
            check_guard_callable(guard.mirror_guard)

        check_guard_callable(guard)

        return guard


def make_compiled_guard(
    specs: list[GuardSpec],
    python_guard: Guard,
) -> Guard:
    """
    Make a guard backed by a C++ guard program.

    The Python guard is deliberately kept as the mirror/oracle for strict
    checking while the hot check path runs inside paddle.framework.core.
    """
    if not ENV_SOT_ENABLE_COMPILED_GUARD.get():
        return python_guard

    cpp_specs: list[tuple[Any, ...]] = []
    for spec in specs:
        if not isinstance(spec, GuardSpec):
            raise UnsupportedCompiledGuard(
                "compiled guard specs must be GuardSpec objects; "
                f"got {type(spec).__name__}: {spec!r}"
            )
        cpp_specs.append(spec.to_cpp_spec())

    compiled_guard = paddle.framework.core.CompiledGuard(cpp_specs)

    def guard(frame):
        return compiled_guard.check(frame)

    guard.expr = compiled_guard.stringify()
    guard.inlined_expr = guard.expr
    guard.compiled_guard = compiled_guard
    guard.original_guard = python_guard
    if ENV_SOT_ENABLE_STRICT_GUARD_CHECK.get():
        guard.mirror_guard = python_guard
    return guard


def make_guard_spec(
    kind: GuardOpKind, access: GuardAccess, *args: Any
) -> GuardSpec:
    if not isinstance(kind, GuardOpKind):
        raise UnsupportedCompiledGuard(
            f"compiled guard kind must be GuardOpKind, got {kind!r}"
        )
    if not access:
        raise UnsupportedCompiledGuard(
            f"{kind.name} guard requires an access path"
        )
    return GuardSpec(kind, access, args)


def support_weak_ref(obj):
    if isinstance(obj, types.FunctionType):
        return True
    return False


def check_guard(
    fn: Callable[[CheckGuardInputT], list[StringifiedExpression]],
) -> Callable[[CheckGuardInputT], list[StringifiedExpression]]:
    def wrapper(self: CheckGuardInputT) -> list[StringifiedExpression]:
        assert self.tracker.is_traceable(), (
            "Cannot make guard from a non-tracable guard variable."
        )

        def guard_log():
            frame_value_tracer = self.tracker.trace_value_from_frame()
            print(
                f"[Guard] guard_fn for {self}, tracker={self.tracker.__class__.__name__}, value={frame_value_tracer.registered_expr}"
            )

        log_do(4, guard_log)
        return fn(self)

    return wrapper


@check_guard
def object_equal_stringified_guard(self) -> list[StringifiedExpression]:
    frame_value_tracer = self.tracker.trace_value_from_frame()

    obj_free_var_name = f"__{self.id}"
    weak_ref_obj = self.get_py_value()
    if support_weak_ref(weak_ref_obj):
        weak_ref_obj = weakref.ref(self.get_py_value())
        return [
            StringifiedExpression(
                f"{obj_free_var_name}() is not None and {{}} == {obj_free_var_name}()",
                [frame_value_tracer],
                union_free_vars(
                    frame_value_tracer.free_vars,
                    {obj_free_var_name: weak_ref_obj},
                ),
            )
        ]
    return [
        StringifiedExpression(
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
