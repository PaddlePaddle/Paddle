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
import types
from typing import TYPE_CHECKING, Any

from paddle._typing import unreached

from ....profiler import EventGuard
from ....utils import do_until_stop_iteration
from ....utils.exceptions import (
    BreakGraphError,
    BreakGraphInlineCallBreak,
    FallbackError,
    FallbackInlineCallBreak,
    OtherInlineCallBreak,
    SotErrorBase,
    UnsupportedOperationBreak,
)
from ..guard import check_faster_guard
from ..tracker import ConstTracker, DanglingTracker, DummyTracker
from .base import (
    VariableBase,
    VariableFactory,
)
from .basic import ConstantVariable
from .callable import BuiltinVariable
from .container import TupleVariable

if TYPE_CHECKING:
    from collections.abc import Sequence

    import paddle

    from ..function_graph import FunctionGraph
    from ..pycode_generator import PyCodeGen
    from ..tracker import Tracker
    from ..virtual_frame import VirtualFrame


class IterVariable(VariableBase):
    """
    This Variable (include subclasses) should be generated only when simulate GET_ITER opcode
    """

    def __init__(self, graph: FunctionGraph, tracker: Tracker):
        super().__init__(graph, tracker)

    def next(self):
        raise NotImplementedError(f"Can not simulate `next` for {type(self)}")

    def to_list(self):
        raise NotImplementedError(
            f"Can not simulate `to_list` for {type(self)}"
        )

    def send(self, value: VariableBase):
        return self.next()

    def get_iter(self):
        return self


class SequenceIterVariable(IterVariable):
    """
    The basic SequenceIterVariable wraps iterators which can be simulated by call getitem
    Currently includes: List | Tuple | Dict (keys) | Range | Tensor | nn.LayerList

    these interfaces is needed:
    - next
    - to_list
    - has_side_effect
    - _reconstruct
    """

    mutable_attrs = ["idx"]

    def __init__(
        self,
        holded: VariableBase | list[VariableBase],
        graph: FunctionGraph,
        tracker: Tracker,
    ):
        if not isinstance(holded, list):
            holded = [holded]
        super().__init__(graph, tracker)
        self.holds = holded
        self.idx = 0
        self.graph.side_effects.record_mutable_variable(self)

    @check_faster_guard
    def make_faster_guard(self) -> list[paddle.framework.core.GuardNode]:
        return [
            guard
            for holded in self.holds
            for guard in holded.make_faster_guard()
        ]

    def make_stringified_guard(self):
        return [
            guard
            for holded in self.holds
            for guard in holded.make_stringified_guard()
        ]

    def next(self):
        holded = self.holds[0]
        if self.idx < len(holded):
            val = holded[self.idx]
            self.idx += 1
            return val
        else:
            raise StopIteration

    def to_list(self) -> list:
        if self.has_side_effect():
            raise FallbackError("Can not convert an used iterator into list")
        holded = self.holds[0]
        self.idx = len(holded)
        retval = []
        for i in range(len(holded)):
            retval.append(holded[i])
        return retval

    def has_side_effect(self) -> bool:
        return self.idx != 0

    def _reconstruct(self, codegen: PyCodeGen):
        if self.has_side_effect():
            super()._reconstruct(codegen)
        else:
            self.holds[0].reconstruct(codegen)
            codegen.gen_get_iter()

    @property
    def main_info(self) -> dict[str, Any]:
        return {
            "idx": self.idx,
        }

    def flatten_inner_vars(self) -> list[VariableBase]:
        holded = self.holds
        return [
            inner_var
            for obj in holded
            for inner_var in obj.flatten_inner_vars()
        ]


class EnumerateVariable(SequenceIterVariable):
    """
    EnumerateVariable holds a SequenceIterVariable and return additional index
    """

    def __init__(
        self, val_iterator: IterVariable, graph: FunctionGraph, tracker: Tracker
    ):
        super().__init__(val_iterator, graph, tracker)

    def next(self):
        val = self.holds[0].next()
        idx_var = ConstantVariable(self.idx, self.graph, ConstTracker(self.idx))
        self.idx += 1
        return TupleVariable(
            (idx_var, val), self.graph, DummyTracker([idx_var, val])
        )

    def to_list(self):
        values = self.holds[0].to_list()
        idx = [
            ConstantVariable(i, self.graph, ConstTracker(i))
            for i in range(len(values))
        ]
        return list(zip(idx, values))

    def has_side_effect(self) -> bool:
        return self.holds[0].has_side_effect()

    def _reconstruct(self, codegen: PyCodeGen):
        if self.has_side_effect():
            super()._reconstruct(codegen)
        else:
            codegen.gen_load_global("enumerate", push_null=True)
            self.holds[0].reconstruct(codegen)
            codegen.gen_call_function(1)

    @staticmethod
    def from_iterator(value, graph: FunctionGraph | None, tracker: Tracker):
        iter_variable = value.get_iter()
        if isinstance(iter_variable, UserDefinedIterVariable):
            return UserDefinedIterVariable(value, graph, tracker)
        else:
            return EnumerateVariable(iter_variable, graph, tracker)


class ZipVariable(SequenceIterVariable):
    """
    ZipVariable holds a list of SequenceIterVariable
    """

    def __init__(
        self, iters: list[IterVariable], graph: FunctionGraph, tracker: Tracker
    ):
        super().__init__(iters, graph, tracker)

    def next(self):
        # can not use <listcomp> here, because it will raise a RuntimeError("StopIteration")
        # but we want a StopIteration Exception
        values = []
        for iter_var in self.holds:
            next_var = iter_var.next()
            values.append(next_var)

        return VariableFactory.from_value(
            tuple(values), self.graph, DummyTracker(values)
        )

    def to_list(self):
        lists = [iter_vars.to_list() for iter_vars in self.holds]
        min_len = min(len(l) for l in lists)
        result = []
        for i in range(min_len):
            result.append(
                VariableFactory.from_value(
                    tuple(l[i] for l in lists),
                    self.graph,
                    DummyTracker(list(self.holds)),
                )
            )
        return result

    def has_side_effect(self) -> bool:
        return any(iter_var.has_side_effect() for iter_var in self.holds)

    def _reconstruct(self, codegen: PyCodeGen):
        if self.has_side_effect():
            super()._reconstruct(codegen)
        else:
            codegen.gen_load_global("zip", push_null=True)
            for iter_var in self.holds:
                iter_var.reconstruct(codegen)
            codegen.gen_call_function(len(self.holds))

    @staticmethod
    def from_iterator(
        value: Sequence[VariableBase],
        graph: FunctionGraph | None,
        tracker: Tracker,
    ):
        assert isinstance(value, (list, tuple))
        zip_targets = []

        for variable in value:
            iter_variable = variable.get_iter()
            if isinstance(iter_variable, UserDefinedIterVariable):
                return UserDefinedIterVariable(value, graph, tracker)
            zip_targets.append(iter_variable)

        return ZipVariable(zip_targets, graph, tracker)


class MapVariable(SequenceIterVariable):
    """
    MapVariable holds a SequenceIterVariable and return a Iterable Variable after map function
    """

    def __init__(self, fn, iters: list[IterVariable], graph, tracker):
        super().__init__(iters, graph, tracker)
        self.fn = fn

    def next(self):
        return self.fn(*[iter_var.next() for iter_var in self.holds])

    def to_list(self) -> list:
        lists = [iter_var.to_list() for iter_var in self.holds]
        min_len = min(len(l) for l in lists)
        result = []
        for i in range(min_len):
            result.append(self.fn(*(l[i] for l in lists)))
        return result

    def has_side_effect(self) -> bool:
        return any(iter_var.has_side_effect() for iter_var in self.holds)

    def _reconstruct(self, codegen: PyCodeGen):
        if self.has_side_effect():
            super()._reconstruct(codegen)
        else:
            codegen.gen_load_global("map", push_null=True)
            self.fn.reconstruct(codegen)
            for iter_var in self.holds:
                iter_var.reconstruct(codegen)
            codegen.gen_call_function(len(self.holds) + 1)

    @staticmethod
    def from_iterator(
        fn,
        value: Sequence[VariableBase],
        graph: FunctionGraph | None,
        tracker: Tracker,
    ):
        map_targets = []

        for variable in value:
            iter_variable = variable.get_iter()
            if isinstance(iter_variable, UserDefinedIterVariable):
                return UserDefinedIterVariable(value, graph, tracker)
            map_targets.append(iter_variable)

        return MapVariable(fn, map_targets, graph, tracker)


class GeneratorVariable(IterVariable):
    def __init__(
        self,
        code_var: VariableBase,
        vframe: VirtualFrame,
        graph: FunctionGraph,
        tracker: Tracker,
    ):
        self.code_var = code_var
        self.vframe = vframe
        self.shared_stack = []
        super().__init__(graph, tracker)

    def send(self, /, value: VariableBase):
        from ..opcode_inline_executor import OpcodeInlineGeneratorExecutor

        checkpoint = self.graph.save_memo()
        frame_state = self.vframe.get_state()
        try:
            inline_gen_executor = OpcodeInlineGeneratorExecutor(
                self.vframe, self.code_var, self.graph
            )
            if sys.version_info < (3, 10) and self.vframe.lasti == 0:
                assert isinstance(value, ConstantVariable)
                assert value.value is None
            else:
                self.vframe.stack.push(value)
            with EventGuard(
                f"Inline Gen Call: {inline_gen_executor.vframe.code.co_name}, file {inline_gen_executor.vframe.code.co_filename}, line {int(inline_gen_executor.vframe.code.co_firstlineno)}"
            ):
                output: VariableBase = inline_gen_executor.inline_call()
                if inline_gen_executor.stop_state == "Return":
                    raise StopIteration
        except SotErrorBase as error:
            self.graph.restore_memo(checkpoint)
            self.vframe.restore_state(frame_state)
            filename = self.code_var.value.co_filename
            lineno = self.code_var.value.co_firstlineno
            code_name = self.code_var.value.co_name
            location_info = f'File "{filename}", line {lineno}, in {code_name}'

            exception_class = OtherInlineCallBreak
            if isinstance(error, BreakGraphError):
                exception_class = BreakGraphInlineCallBreak
            elif isinstance(error, FallbackError):
                exception_class = FallbackInlineCallBreak

            raise BreakGraphError(
                exception_class(
                    f"{location_info} encountered breakgraph error caused by\n    {error}"
                )
            )

        return output

    def getattr(self, name: str, default=None):
        from ..dispatch_functions import generator_send

        known_generator_attrs = {"send"}
        if name not in known_generator_attrs:
            raise BreakGraphError(
                UnsupportedOperationBreak(
                    reason_str=f"Get attribute {name} from generator is not supported."
                )
            )
        if name == "send":
            return BuiltinVariable(
                generator_send, self.graph, DanglingTracker()
            ).bind_dangling_fn(self, "send")
        unreached()

    def get_py_value(self, allow_tensor=False):
        raise BreakGraphError(
            UnsupportedOperationBreak(
                reason_str="Get real value from generator is not supported."
            )
        )

    def get_py_type(self):
        return types.GeneratorType

    def next(self):
        return self.send(ConstantVariable.wrap_literal(None, self.graph))

    def to_list(self):
        return do_until_stop_iteration(lambda: self.next())

    @property
    def main_info(self) -> dict[str, Any]:
        return {
            "co_name": self.code_var.value.co_name,
        }

    # @VariableFactory.register_from_value()
    # def from_value(value: Any, graph: FunctionGraph, tracker: Tracker):
    #     if inspect.isgenerator(value):
    #         return GeneratorVariable()
    #     return None


# what UserDefinedIterVariable holds doesn't matter, because use user defined iterator will trigger break graph
class UserDefinedIterVariable(IterVariable):
    def __init__(
        self,
        holded: VariableBase | list[VariableBase],
        graph: FunctionGraph,
        tracker: Tracker,
    ):
        if not isinstance(holded, list):
            holded = [holded]
        self.holds = holded
        super().__init__(graph, tracker)

    def to_list(self):
        raise BreakGraphError(
            UnsupportedOperationBreak(
                reason_str="Break graph when iterating user defined iterator"
            )
        )

    def next(self):
        raise BreakGraphError(
            UnsupportedOperationBreak(
                reason_str="Break graph when iterating user defined iterator"
            )
        )

    @check_faster_guard
    def make_faster_guard(self) -> list[paddle.framework.core.GuardNode]:
        return [
            guard
            for holded in self.holds
            for guard in holded.make_faster_guard()
        ]

    def make_stringified_guard(self):
        return [
            guard
            for holded in self.holds
            for guard in holded.make_stringified_guard()
        ]
