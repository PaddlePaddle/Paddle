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

import builtins
from typing import TYPE_CHECKING

from ...utils import InnerError, NameGenerator
from .guard import StringifyExpression, union_free_vars

if TYPE_CHECKING:
    from .pycode_generator import PyCodeGen
    from .variables import VariableBase


class Tracker:
    inputs: list[VariableBase]
    name_generator = NameGenerator("tracker_")

    def __init__(self, inputs: list[VariableBase]):
        self.inputs = inputs
        self.id = Tracker.name_generator.next()

    def gen_instructions(self, codegen: PyCodeGen):
        raise NotImplementedError()

    def trace_value_from_frame(self) -> StringifyExpression:
        raise NotImplementedError()

    def is_traceable(self):
        for input in self.inputs:
            if not input.tracker.is_traceable():
                return False
        return True


class DummyTracker(Tracker):
    def __init__(self, inputs: list[VariableBase]):
        super().__init__(inputs)

    def gen_instructions(self, codegen: PyCodeGen):
        raise InnerError("DummyTracker has no instructions")

    def trace_value_from_frame(self):
        raise InnerError("DummyTracker can't trace value from frame")

    def is_traceable(self):
        return False

    def __repr__(self) -> str:
        return f"DummyTracker(num_inputs={len(self.inputs)})"


class LocalTracker(Tracker):
    def __init__(self, name: str):
        super().__init__([])
        self.name = name

    def gen_instructions(self, codegen: PyCodeGen):
        codegen.gen_load_fast(self.name)

    def trace_value_from_frame(self):
        return StringifyExpression(f"frame.f_locals['{self.name}']", {})

    def __repr__(self) -> str:
        return f"LocalTracker(name={self.name})"


class GlobalTracker(Tracker):
    def __init__(self, name):
        super().__init__([])
        self.name = name

    def gen_instructions(self, codegen: PyCodeGen):
        codegen.gen_load_global(self.name)

    def trace_value_from_frame(self):
        return StringifyExpression(f"frame.f_globals['{self.name}']", {})

    def __repr__(self) -> str:
        return f"GlobalTracker(name={self.name})"


class BuiltinTracker(Tracker):
    def __init__(self, name: str):
        super().__init__([])
        self.name = name

    def gen_instructions(self, codegen: PyCodeGen):
        codegen.gen_load_global(self.name)

    def trace_value_from_frame(self):
        return StringifyExpression(
            f"builtins.__dict__[{self.name}]", {"builtins": builtins}
        )

    def __repr__(self) -> str:
        return f"BuiltinTracker(name={self.name})"


class ConstTracker(Tracker):
    def __init__(self, value):
        super().__init__([])
        self.value = value

    def gen_instructions(self, codegen: PyCodeGen):
        codegen.gen_load_const(self.value)

    def trace_value_from_frame(self):
        return StringifyExpression(f"{self.value}", {})

    def __repr__(self) -> str:
        return f"ConstTracker(value={self.value})"


class GetAttrTracker(Tracker):
    def __init__(self, obj: VariableBase, attr: str):
        super().__init__([obj])
        self.obj = obj
        self.attr = attr

    def gen_instructions(self, codegen: PyCodeGen):
        self.obj.tracker.gen_instructions(codegen)
        codegen.gen_load_attr(self.attr)

    def trace_value_from_frame(self):
        obj_tracer = self.obj.tracker.trace_value_from_frame()
        if self.attr.isidentifier():
            expr = f"{obj_tracer.expr}.{self.attr}"
        else:
            expr = f"getattr({obj_tracer.expr}, '{self.attr}')"
        return StringifyExpression(
            expr,
            union_free_vars(obj_tracer.free_vars),
        )

    def __repr__(self) -> str:
        return f"GetAttrTracker(attr={self.attr})"


class GetItemTracker(Tracker):
    def __init__(self, container_var: VariableBase, key: object):
        super().__init__([container_var])
        self.container = container_var
        self.key = key

    def gen_instructions(self, codegen: PyCodeGen):
        self.container.tracker.gen_instructions(codegen)
        codegen.gen_load_const(self.key)
        codegen.gen_subscribe()

    def trace_value_from_frame(self):
        container_tracer = self.container.tracker.trace_value_from_frame()
        return StringifyExpression(
            f"{container_tracer.expr}[{self.key!r}]",
            union_free_vars(container_tracer.free_vars),
        )

    def __repr__(self) -> str:
        return f"GetItemTracker(key={self.key!r})"
