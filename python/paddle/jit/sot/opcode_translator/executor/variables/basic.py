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

import operator
import types
from functools import cached_property, reduce
from typing import TYPE_CHECKING, Any

import numpy as np

import paddle
from paddle.framework import core

from ....infer_meta import MetaInfo
from ....symbolic.statement_ir import Symbol
from ....utils import (
    ENV_SOT_ALLOW_DYNAMIC_SHAPE,
    BreakGraphError,
    ConstTypes,
    FallbackError,
    NameGenerator,
    paddle_tensor_methods,
    printable,
)
from ....utils.exceptions import HasNoAttributeError, InnerError
from ..dispatch_functions import tensor_numel
from ..guard import (
    StringifyExpression,
    check_guard,
    object_equal_stringify_guard,
    stringify_pyobject,
    union_free_vars,
)
from ..mutable_data import MutableDictLikeData
from ..pycode_generator import PyCodeGen
from ..tracker import (
    ConstTracker,
    DanglingTracker,
    DummyTracker,
    GetAttrTracker,
    GetIterTracker,
    GlobalTracker,
    SymbolicOperationTracker,
    Tracker,
)
from .base import VariableBase, VariableFactory

if TYPE_CHECKING:
    from ..function_graph import FunctionGraph
    from .callable import FunctionVariable


FP_DTYPE_ABBRS = {
    core.DataType.BFLOAT16: "bfloat16",
    core.DataType.FLOAT64: "float64",
    core.DataType.FLOAT32: "float32",
    core.DataType.FLOAT16: "float16",
}

CP_DTYPE_ABBRS = {
    core.DataType.COMPLEX64: "complex64",
    core.DataType.COMPLEX128: "complex128",
}

INT_DTYPE_ABBRS = {
    core.DataType.INT8: "int8",
    core.DataType.INT16: "int16",
    core.DataType.INT32: "int32",
    core.DataType.INT64: "int64",
    core.DataType.UINT8: "uint8",
}

DTYPE_ABBRS = {
    **FP_DTYPE_ABBRS,
    **CP_DTYPE_ABBRS,
    **INT_DTYPE_ABBRS,
    core.DataType.BOOL: "bool",
}


class ConstantVariable(VariableBase):
    """
    ConstantVariable is a subclass of VariableBase used to wrap a Variable of the const type.

    Args:
        value(Any): The value to be wrapped.
        tracker(Tracker): The Tracker object that tracks the information of this variable.
    """

    def __init__(
        self,
        value: Any,
        graph: FunctionGraph,
        tracker: Tracker,
    ):
        super().__init__(graph, tracker)
        self.value = value

    def get_py_value(self, allow_tensor=False):
        return self.value

    @property
    def debug_name(self) -> str:
        return f"{self.value}"

    @debug_name.setter
    def debug_name(self, name):
        pass

    def _reconstruct(self, codegen: PyCodeGen):
        codegen.gen_load_const(self.value)

    @property
    def main_info(self) -> dict[str, Any]:
        return {"value": self.value}

    def __bool__(self) -> bool:
        return bool(self.value)

    def bool(self):
        return ConstantVariable(bool(self), self.graph, DummyTracker([self]))

    def bool_not(self):
        assert isinstance(
            self.get_py_value(), bool
        ), "Bool_not can only be applied to a bool variable."
        return ConstantVariable(
            not bool(self.get_py_value()), self.graph, DummyTracker([self])
        )

    def str(self):
        return ConstantVariable(
            str(self.value), self.graph, DummyTracker([self])
        )

    def format(self, *args):
        return ConstantVariable(
            str(self.value).format(*[str(a.value) for a in args]),
            self.graph,
            DummyTracker([self, *args]),
        )

    def lower(self):
        return ConstantVariable(
            str(self.value).lower(),
            self.graph,
            DummyTracker([self]),
        )

    def ord(self):
        return ConstantVariable(
            ord(self.value),
            self.graph,
            DummyTracker([self]),
        )

    def chr(self):
        return ConstantVariable(
            chr(self.value),
            self.graph,
            DummyTracker([self]),
        )

    @check_guard
    def make_stringify_guard(self) -> list[StringifyExpression]:
        if (
            ENV_SOT_ALLOW_DYNAMIC_SHAPE.get()
            and isinstance(self.value, int)
            and self.tracker.need_guard()
        ):
            from ..executor_cache import OpcodeExecutorCache

            frame_value_tracer = self.tracker.trace_value_from_frame()
            symbolic_inputs = OpcodeExecutorCache().symbolic_inputs
            symbolic_inputs.setdefault(frame_value_tracer.inlined_expr, {})
            symbolic_input = symbolic_inputs[frame_value_tracer.inlined_expr]
            symbolic_input.setdefault(self.value, 0)
            symbolic_input[self.value] += 1

        return super().make_stringify_guard()

    @VariableFactory.register_from_value()
    def from_value(value: Any, graph: FunctionGraph, tracker: Tracker):
        if type(value) in ConstTypes:
            return ConstantVariable(value, graph, tracker)
        return None

    @staticmethod
    def wrap_literal(value: Any, graph: FunctionGraph) -> ConstantVariable:
        """
        Wrap a literal value in a ConstantVariable.

        Args:
            value(Any): The literal value to be wrapped.

        Returns:
            ConstantVariable: A new ConstantVariable object that wraps the given value.
        """
        if isinstance(value, ConstantVariable):
            return value
        assert isinstance(
            value, ConstTypes
        ), f"value: {value},type: {type(value)}"
        return ConstantVariable(value, graph, ConstTracker(value))


class PrintStmtVariable(VariableBase):
    def __init__(self, value: Any, graph: FunctionGraph):
        # TODO: graph should be not None
        super().__init__(None, DanglingTracker())
        self.args, self.kwargs = value
        self.graph = graph

    def _reconstruct(self, codegen: PyCodeGen):
        # do we need ? may be too strict.
        for var in self.args:
            self.graph.add_global_guarded_variable(var)
        for var in self.kwargs.values():
            self.graph.add_global_guarded_variable(var)
        # currently don't consider kwargs
        codegen.gen_load_global("print", push_null=True)
        for var in self.args:
            var.reconstruct(codegen)
        codegen.gen_call_function(len(self.args))
        codegen.gen_pop_top()

    def flatten_items(self):
        return self.args


IMPLEMENTED_TENSOR_PROPERTIES = set()


def tensor_property(func):
    IMPLEMENTED_TENSOR_PROPERTIES.add(func.__name__)
    return property(func)


class DataVariable(VariableBase):
    """
    A value only object.
    If it's all magic method don't change the function_graph state, [tensor op, guard, side_effect]
    we will call it a ValueObjectVariable, we directly call python operator on it.
    """

    def __init__(
        self,
        value: Any,
        graph: FunctionGraph,
        tracker: Tracker,
    ):
        super().__init__(graph, tracker)
        self.value = value

    def get_py_value(self, allow_tensor=False):
        return self.value


class TensorDtypeVariable(DataVariable):
    def __init__(self, value, graph, tracker):
        super().__init__(value, graph, tracker)

    @check_guard
    def make_stringify_guard(self) -> list[StringifyExpression]:
        if isinstance(self.tracker, GetAttrTracker) and isinstance(
            self.tracker.obj, TensorVariable
        ):
            tensor_value_tracer = (
                self.tracker.obj.tracker.trace_value_from_frame()
            )
            dtype_str, dtype_free_vars = stringify_pyobject(self.value)
            return [
                StringifyExpression(
                    f"MetaInfo.from_tensor({{}}).dtype == {dtype_str}",
                    [tensor_value_tracer],
                    union_free_vars(
                        {"MetaInfo": MetaInfo},
                        dtype_free_vars,
                    ),
                )
            ]
        else:
            return object_equal_stringify_guard(self)

    def get_py_value(self, allow_tensor=False):
        return super().get_py_value(allow_tensor)

    def get_py_type(self):
        return super().get_py_type()

    def _reconstruct(self, codegen: PyCodeGen):
        # dtype of paddle.Tensor is hashable, we can just load it as const var
        codegen.gen_load_const(self.value)

    @property
    def main_info(self) -> dict[str, Any]:
        return {
            "dtype": self.value,
        }

    @VariableFactory.register_from_value()
    def from_value(value: Any, graph: FunctionGraph, tracker: Tracker):
        if isinstance(
            value, (paddle.core.VarDesc.VarType, paddle.core.DataType)
        ):
            return TensorDtypeVariable(value, graph, tracker)


class TensorVariable(VariableBase):
    """
    TensorVariable is a subclass of VariableBase used to wrap a Variable of the tensor type.

    Args:
        tensor (paddle.Tensor | MetaInfo): The tensor to be wrapped.
        graph (FunctionGraph): The FunctionGraph object that this variable is associated with.
        tracker (Tracker): The Tracker object that tracks the information of this variable.
    """

    var_name_generator = NameGenerator("var_")
    mutable_attrs = ["meta"]

    def __init__(
        self,
        tensor: paddle.Tensor | MetaInfo,
        graph: FunctionGraph,
        tracker: Tracker,
    ):
        super().__init__(graph, tracker)
        if isinstance(tensor, paddle.Tensor):
            self.value = None
            self.meta = MetaInfo.from_tensor(tensor)
        elif isinstance(tensor, MetaInfo):
            self.value = None
            self.meta = tensor
        else:
            raise InnerError(
                f"Required type(tensor) is paddle.Tensor or ProxyTensor, but received {type(tensor).__name__}."
            )
        self.origin_meta = self.meta
        self.var_name = TensorVariable.var_name_generator.next()
        self.graph.side_effects.record_mutable_variable(self)

    def __len__(self):
        if self.meta.shape[0] == -1:
            raise BreakGraphError(
                "length of tensor variable with first dimension == -1"
            )
        return self.meta.shape[0]

    def get_py_value(self, allow_tensor=False):
        if allow_tensor:

            class SotTensor:
                def __init__(self, id_):
                    self.id = id_

                def __eq__(self, var):
                    if not hasattr(var, "id"):
                        return False
                    else:
                        return self.id == var.id

                def __hash__(self):
                    return hash(self.id)

            return SotTensor(self.id)

        raise BreakGraphError(
            "Called TensorVariable.get_py_value. Should not use Tensor's value in simulating."
        )

    def get_py_type(self):
        return paddle.Tensor

    def get_symbol(self) -> Symbol:
        return Symbol(self.var_name)

    @property
    def out_var_name(self):
        return f"{self.graph.OUT_VAR_PREFIX}{self.var_name}"

    def _reconstruct(self, codegen: PyCodeGen):
        codegen.gen_load_fast(self.out_var_name)

    @check_guard
    def make_stringify_guard(self) -> list[StringifyExpression]:
        frame_value_tracer = self.tracker.trace_value_from_frame()

        return [
            StringifyExpression(
                f"MetaInfo.from_tensor({{}}).guard_str() == '{self.origin_meta.guard_str()}'",
                [frame_value_tracer],
                union_free_vars(
                    {"MetaInfo": MetaInfo},
                    frame_value_tracer.free_vars,
                ),
            )
        ]

    def get_iter(self):
        from .iter import SequenceIterVariable

        return SequenceIterVariable(self, self.graph, GetIterTracker(self))

    @property
    def main_info(self) -> dict[str, Any]:
        dtype = self.meta.dtype
        if isinstance(dtype, paddle.core.VarDesc.VarType):
            dtype = paddle.pir.core.vartype_to_datatype[dtype]
        return {
            "shape": self.meta.shape,
            "dtype": DTYPE_ABBRS[dtype],
            "stop_gradient": self.meta.stop_gradient,
            "var_name": self.var_name,
        }

    def getitem(self, key):
        return self.graph.call_tensor_method("__getitem__", self, key)

    def setitem(self, key, value):
        self.graph.add_global_guarded_variable(value)

        key_var = VariableFactory.from_value(
            key, self.graph, tracker=ConstTracker(key)
        )
        new_tensor = self.graph.call_paddle_api(
            paddle.static.setitem,
            self,
            key_var,
            value,
        )

        self.meta = new_tensor.meta
        self.graph.add_inplace_tensors(self)

    @tensor_property
    def T(self):
        """
        Return a new TensorVariable object that wraps the result of calling the transpose method on the wrapped value of this TensorVariable.
        """
        from .container import ListVariable

        perm = list(range(len(self.meta.shape) - 1, -1, -1))
        perm_var = ListVariable(perm, self.graph, tracker=ConstTracker(perm))
        assert perm_var is not None
        out = self.graph.call_paddle_api(paddle.transpose, self, perm_var)
        return out

    @tensor_property
    def ndim(self):
        """
        Return a ConstantVariable object that represents the number of dimensions of the wrapped value of this TensorVariable.
        """
        return ConstantVariable(
            len(self.meta.shape), self.graph, DummyTracker([self])
        )

    @tensor_property
    def size(self):
        """
        Return a ConstantVariable object that represents the total number of elements in the wrapped value of this TensorVariable.
        """
        # TODO: maybe break graph.
        if self.meta.is_dynamic_shape():
            raise BreakGraphError(
                f"Getting size for a dynamic shape tensor causes graph break. shape = {self.meta.shape}"
            )
        elements = reduce(operator.mul, self.meta.shape, 1)
        return ConstantVariable(elements, self.graph, DummyTracker([self]))

    @tensor_property
    def shape(self):
        if self.meta.is_dynamic_shape():
            raise BreakGraphError(
                f"Getting shape for a dynamic shape tensor causes graph break. shape = {self.meta.shape}"
            )
        from .container import ListVariable

        return ListVariable(
            self.meta.shape, self.graph, tracker=DummyTracker([self])
        )

    def numel(self):
        return self.size

    def len(self):
        if len(self.meta.shape) == 0:
            raise InnerError("len() of a 0-D tensor is wrong")
        first_dim = self.meta.shape[0]
        if first_dim == -1:
            raise BreakGraphError(
                "Getting len() for a dynamic shape tensor causes graph break."
            )

        return ConstantVariable(first_dim, self.graph, DummyTracker([self]))

    def is_tensor(self):
        return ConstantVariable(True, self.graph, DummyTracker([self]))

    def is_complex(self):
        dtype = self.meta.dtype
        if isinstance(dtype, paddle.core.VarDesc.VarType):
            dtype = paddle.pir.core.vartype_to_datatype[dtype]
        is_cp_dtype = dtype in CP_DTYPE_ABBRS
        return ConstantVariable(is_cp_dtype, self.graph, DummyTracker([self]))

    def is_integer(self):
        dtype = self.meta.dtype
        if isinstance(dtype, paddle.core.VarDesc.VarType):
            dtype = paddle.pir.core.vartype_to_datatype[dtype]
        is_int_dtype = dtype in INT_DTYPE_ABBRS
        return ConstantVariable(is_int_dtype, self.graph, DummyTracker([self]))

    def is_floating_point(self):
        dtype = self.meta.dtype
        if isinstance(dtype, paddle.core.VarDesc.VarType):
            dtype = paddle.pir.core.vartype_to_datatype[dtype]
        is_fp_dtype = dtype in FP_DTYPE_ABBRS
        return ConstantVariable(is_fp_dtype, self.graph, DummyTracker([self]))

    def getattr(self, name: str, default=None):
        if default is not None:
            raise FallbackError(
                "default argument for getattr is not implemented"
            )
        method_name_to_builtin_fn = {
            "dim": paddle.rank,
            "numel": tensor_numel,
            "ndimension": paddle.rank,
            "is_tensor": paddle.is_tensor,
            "is_complex": paddle.is_complex,
            "is_integer": paddle.is_integer,
            "is_floating_point": paddle.is_floating_point,
        }
        if name in ["dtype", "type", "name", "persistable", "stop_gradient"]:
            if name == "name" and self.meta.name.startswith(
                "infer_meta_variable_tmp"
            ):
                raise BreakGraphError(f"{self.meta.name} is a middle tensor.")
            return VariableFactory.from_value(
                getattr(self.meta, name),
                self.graph,
                tracker=GetAttrTracker(self, name),
            )
        elif name in IMPLEMENTED_TENSOR_PROPERTIES:
            return getattr(self, name)
        elif name in method_name_to_builtin_fn:
            # TODO: backward, gradient
            from .callable import BuiltinVariable

            builtin_fn = method_name_to_builtin_fn[name]

            return BuiltinVariable(
                builtin_fn, self.graph, DanglingTracker()
            ).bind(self, name)
        elif name in paddle_tensor_methods:
            from .callable import TensorFunctionVariable

            fn_var = TensorFunctionVariable(
                name, graph=self.graph, tracker=DanglingTracker()
            )
            return fn_var.bind(self, name)
        else:
            raise HasNoAttributeError(f"Unknown Tensor attribute: {name}")

    def setattr(self, key, val):
        # support tensor variable store attr, like:
        # t.stop_gradient = True
        self.graph.call_tensor_method(
            "__setattr__",
            self,
            VariableFactory().from_value(key, self.graph, ConstTracker(key)),
            val,
        )

    def delattr(self, key):
        raise BreakGraphError("Don't support TensorVariable delattr")

    @VariableFactory.register_from_value()
    def from_value(value: Any, graph: FunctionGraph, tracker: Tracker):
        if isinstance(value, (paddle.Tensor, MetaInfo)):
            return TensorVariable(value, graph, tracker)
        return None


class SymbolicVariable(VariableBase):
    """
    TODO
    """

    var_name_generator = NameGenerator("symint_")

    def __init__(
        self,
        value: int | MetaInfo,
        graph: FunctionGraph,
        tracker: Tracker,
    ):
        super().__init__(graph, tracker)
        self.var_name = self.var_name_generator.next()
        if isinstance(value, MetaInfo):
            self.value = None
            self.meta = value
        else:
            self.value = value
            self.meta = MetaInfo(
                [], paddle.int64, True, self.var_name, False, None, None
            )
        self.need_guard_value = False

    def get_py_value(self, allow_tensor=False):
        self.need_guard_value = True
        if self.value is None:
            assert isinstance(
                self.tracker, SymbolicOperationTracker
            ), f"self.value is None, but tracker is not SymbolicOperationTracker. tracker: {self.tracker}"
            inputs = self.tracker.inputs
            assert len(inputs) >= 1
            other_inputs_value = [x.get_py_value() for x in inputs[1:]]
            self.value = getattr(
                inputs[0].get_py_value(), self.tracker.method_name
            )(*other_inputs_value)
        return self.value

    def get_py_type(self):
        # TODO(zrr1999): not need to use value to get type
        return super().get_py_type()

    def get_symbol(self) -> Symbol:
        return Symbol(self.var_name)

    def __bool__(self) -> bool:
        return bool(self.get_py_value())

    def bool(self):
        return ConstantVariable(bool(self), self.graph, DummyTracker([self]))

    @property
    def out_var_name(self):
        return f"{self.graph.OUT_VAR_PREFIX}{self.var_name}"

    def _reconstruct(self, codegen: PyCodeGen):
        codegen.gen_load_fast(self.out_var_name)
        codegen.gen_load_method("item")
        codegen.gen_call_method(0)  # TODO

    @check_guard
    def make_stringify_guard(self) -> list[StringifyExpression]:
        assert ENV_SOT_ALLOW_DYNAMIC_SHAPE.get()
        from ..executor_cache import OpcodeExecutorCache

        frame_value_tracer = self.tracker.trace_value_from_frame()
        symbolic_inputs = OpcodeExecutorCache().symbolic_inputs

        assert frame_value_tracer.inlined_expr in symbolic_inputs

        # TODO(zrr1999): Once dynamic shape is used, there will be no new guards
        symbolic_input = symbolic_inputs[frame_value_tracer.inlined_expr]
        symbolic_input.setdefault(self.value, 0)
        symbolic_input[self.value] += 1
        if self.need_guard_value:
            return super().make_stringify_guard()
        return [
            StringifyExpression(
                f"id(type({{}})) == {id(self.get_py_type())}",
                [frame_value_tracer],
                union_free_vars(frame_value_tracer.free_vars),
            )
        ]

    @VariableFactory.register_from_value(successor="ConstantVariable")
    def from_value(value: Any, graph: FunctionGraph, tracker: Tracker):
        if not ENV_SOT_ALLOW_DYNAMIC_SHAPE.get():
            return
        if not isinstance(value, int):
            return
        if not tracker.need_guard():
            return

        from ..executor_cache import OpcodeExecutorCache

        symbolic_inputs = OpcodeExecutorCache().symbolic_inputs

        for tracker_expr, symbolic_input in symbolic_inputs.items():
            if tracker.match_expr(tracker_expr):
                symbolic_input.setdefault(value, 0)
                symbolic_input[value] += 1
                # TODO(zrr1999): determine frequency
                return SymbolicVariable(value, graph, tracker)
        return None


class ParameterVariable(TensorVariable):
    def __init__(
        self,
        param: paddle.Tensor | MetaInfo,
        graph: FunctionGraph,
        tracker: Tracker,
    ):
        super().__init__(param, graph, tracker)

    @VariableFactory.register_from_value(successor="TensorVariable")
    def from_value(value: Any, graph: FunctionGraph, tracker: Tracker):
        if isinstance(value, (paddle.base.framework.EagerParamBase)):
            return ParameterVariable(value, graph, tracker)
        return None


class ObjectVariable(VariableBase):
    """
    ObjectVariable is a subclass of VariableBase used to wrap a Variable of the object type.

    Args:
        obj(Any): The object to be wrapped.
        graph(FunctionGraph): The FunctionGraph object that this variable is associated with.
        tracker(Tracker): The Tracker object that tracks the information of this variable.
    """

    make_stringify_guard = object_equal_stringify_guard

    def __init__(self, obj, graph, tracker):
        super().__init__(graph, tracker)
        self.value = obj

    @property
    def main_info(self) -> dict[str, Any]:
        # NOTE(SigureMo): There are some objects that cannot be printed, such as
        # uninitialized dataclass, we should fallback to the class name.
        if printable(self.value):
            return {"value": self.value}
        else:
            return {"value": f"instance {self.value.__class__.__name__}"}

    def get_py_value(self, allow_tensor=False) -> Any:
        return self.value


class SliceVariable(VariableBase):
    """
    SliceVariable is a subclass of VariableBase used to wrap a Variable of the slice type.

    Args:
        slice_(slice): The slice to be wrapped.
        graph(FunctionGraph): The FunctionGraph object that this variable is associated with.
        tracker(Tracker): The Tracker object that tracks the information of this variable.
    """

    def __init__(self, slice_: slice, graph, tracker):
        super().__init__(graph, tracker)
        self.value = slice_

    @property
    def debug_name(self) -> str:
        return ":".join(
            [
                str(self.value.start) if self.value.start is not None else "",
                str(self.value.stop) if self.value.stop is not None else "",
                str(self.value.step) if self.value.step is not None else "",
            ]
        )

    @debug_name.setter
    def debug_name(self, name):
        pass

    @cached_property
    def attr_proxy(self):
        return self.graph.side_effects.get_proxy(
            MutableDictLikeData, self.value, self.attr_proxy_getter
        )

    @property
    def main_info(self) -> dict[str, Any]:
        return {"value": self.value}

    def get_py_value(self, allow_tensor=False):
        return slice(
            self.getattr("start").get_py_value(),
            self.getattr("stop").get_py_value(),
            self.getattr("step").get_py_value(),
        )

    @check_guard
    def make_stringify_guard(self) -> list[StringifyExpression]:
        frame_value_tracer = self.tracker.trace_value_from_frame()
        result = (
            [
                StringifyExpression(
                    "isinstance({}, slice)",
                    [frame_value_tracer],
                    frame_value_tracer.free_vars,
                ),
            ]
            + self.getattr("start").make_stringify_guard()
            + self.getattr("stop").make_stringify_guard()
            + self.getattr("step").make_stringify_guard()
        )
        return result

    def _reconstruct(self, codegen: PyCodeGen):
        if all(
            isinstance(x, ConstantVariable)
            for x in [
                self.getattr("start"),
                self.getattr("stop"),
                self.getattr("step"),
            ]
        ):
            self.graph.add_global_guarded_variable(self)
            self.getattr("start").reconstruct(codegen)
            self.getattr("stop").reconstruct(codegen)
            self.getattr("step").reconstruct(codegen)
            codegen.gen_build_slice(3)
        else:
            super()._reconstruct(codegen)

    def setattr(self, key, val):
        raise BreakGraphError("Don't support SliceVariable setattr")

    def delattr(self, key):
        raise BreakGraphError("Don't support SliceVariable delattr")

    @VariableFactory.register_from_value()
    def from_value(value: Any, graph: FunctionGraph, tracker: Tracker):
        if isinstance(value, slice):
            return SliceVariable(value, graph, tracker)
        return None


class ModuleVariable(VariableBase):
    """
    ModuleVariable is a subclass of VariableBase used to wrap a Variable of the module type.

    Args:
        func: The module to be wrapped.
        graph: The FunctionGraph object that this variable is associated with.
        tracker: The Tracker object that tracks the information of this variable.
    """

    def __init__(self, func, graph, tracker):
        super().__init__(graph, tracker)
        self.value = func

    def get_py_value(self, allow_tensor=False):
        return self.value

    @property
    def main_info(self) -> dict[str, Any]:
        return {"value": self.value}

    @VariableFactory.register_from_value()
    def from_value(value: Any, graph: FunctionGraph, tracker: Tracker):
        if isinstance(value, types.ModuleType):
            return ModuleVariable(value, graph, tracker)
        return None

    # Happened in a inline import statement.
    make_stringify_guard = object_equal_stringify_guard


class DygraphTracerVariable(VariableBase):
    # TODO(SigureMo): Remove this trick after we add CompareTracker
    def __init__(self, value, graph, tracker):
        super().__init__(graph, tracker)
        self.value = value

    def get_py_value(self, allow_tensor=False):
        return self.value

    @check_guard
    def make_stringify_guard(self) -> list[StringifyExpression]:
        return []

    @property
    def main_info(self) -> dict[str, Any]:
        return {
            "is_none": self.value is None,
        }

    @VariableFactory.register_from_value()
    def from_value(value: Any, graph: FunctionGraph, tracker: Tracker):
        if isinstance(value, paddle.base.dygraph.tracer.Tracer):
            return DygraphTracerVariable(value, graph, tracker)
        return None


class NumpyVariable(VariableBase):
    """
    NumpyVariable is a subclass of VariableBase used to wrap a Variable of the numpy type.

    Args:
        value: The numpy value to be wrapped.
        graph: The FunctionGraph object that this variable is associated with.
        tracker: The Tracker object that tracks the information of this variable.
    """

    def __init__(self, value, graph, tracker):
        super().__init__(graph, tracker)
        self.value = value

    @property
    def main_info(self) -> dict[str, Any]:
        return {"value": self.value}

    def get_py_value(self, allow_tensor=False) -> Any:
        return self.value

    @check_guard
    def make_stringify_guard(self) -> list[StringifyExpression]:
        if isinstance(self.get_py_value(), np.number):
            frame_value_tracer = self.tracker.trace_value_from_frame()

            def format_dtype(dtype: np.dtype):
                return f"np.{str(dtype)}"

            def format_number(number: np.number):
                return f"{format_dtype(number.dtype)}({str(number.item())})"

            return [
                StringifyExpression(
                    f"{{}} == {format_number(self.get_py_value())}",
                    [frame_value_tracer],
                    union_free_vars(frame_value_tracer.free_vars, {"np": np}),
                ),
                StringifyExpression(
                    f"{{}}.dtype == {format_dtype(self.get_py_value().dtype)}",
                    [frame_value_tracer],
                    union_free_vars(frame_value_tracer.free_vars, {"np": np}),
                ),
            ]
        else:
            return object_equal_stringify_guard(self)

    @VariableFactory.register_from_value()
    def from_value(value: Any, graph: FunctionGraph, tracker: Tracker):
        if isinstance(value, (np.ndarray, np.number)):
            return NumpyVariable(value, graph, tracker)
        return None


class NullVariable(VariableBase):
    """
    NullVariable is a subclass of VariableBase used to represent a placeholder variable that has no value or reference associated with it.
    """

    def __init__(self):
        # TODO: graph should be not None
        super().__init__(None, DanglingTracker())

    def __call__(self, *args, **kwargs):
        func = args[0]
        assert callable(func)
        return func(*args[1:], **kwargs)

    def reconstruct(self, codegen: PyCodeGen):
        codegen.gen_load_null_variable()


class CellVariable(VariableBase):
    def __init__(self, value=None):
        # TODO: graph should be not None
        super().__init__(
            None, DanglingTracker()
        )  # should reconstruct cell variable
        assert isinstance(value, (VariableBase, type(None)))
        self.set_value(value)

    def reconstruct(
        self,
        codegen: PyCodeGen,
        *,
        use_tracker: bool = True,
        add_to_global_guarded_vars: bool = True,
    ):
        raise FallbackError("Break graph in closure is not support.")

    def cell_content(self):
        return self.value

    def set_value(self, value):
        self.value = value

    def empty(self):
        return self.value is None


class GlobalVariable(VariableBase):
    def __init__(
        self,
        val_dict,
        graph: FunctionGraph,
        tracker: Tracker,
    ):
        super().__init__(graph, tracker)
        self.proxy = self.graph.side_effects.get_proxy(
            MutableDictLikeData, val_dict, self.proxy_getter
        )

    def proxy_getter(self, proxy: MutableDictLikeData, key: Any):
        if key not in proxy.original_data:
            return MutableDictLikeData.Empty()
        return VariableFactory.from_value(
            proxy.original_data[key],
            self.graph,
            tracker=GlobalTracker(key),
        )

    def get_value(self):
        return dict(self.proxy.get_all().items())

    def keys(self):
        return self.proxy.get_all().keys()

    def get(self, key):
        if isinstance(key, VariableBase):
            raise InnerError(
                f"[{self.__class__.__name__}]: received {key} to get value."
            )
        return self.proxy.get(key)

    def set(self, key, value):
        if isinstance(key, VariableBase):
            raise InnerError(
                f"[{self.__class__.__name__}]: received {key} as key."
            )
        if not isinstance(value, VariableBase):
            raise InnerError(
                f"[{self.__class__.__name__}]: received {value} to set value."
            )
        self.proxy.set(key, value)
        self.graph.side_effects.record_proxy_variable(self)

    def delete(self, key):
        self.proxy.delete(key)
        self.graph.side_effects.record_proxy_variable(self)


class FunctionGlobalVariable(GlobalVariable):
    def __init__(
        self,
        fn: FunctionVariable,
        val_dict: dict[str, Any],
        graph: FunctionGraph,
        tracker: Tracker,
    ):
        super().__init__(val_dict, graph, tracker)
        self.fn = fn

    def proxy_getter(self, proxy: MutableDictLikeData, key: Any):
        from ..opcode_inline_executor import FunctionGlobalTracker

        if key not in proxy.original_data:
            return MutableDictLikeData.Empty()
        return VariableFactory.from_value(
            proxy.original_data[key],
            self.graph,
            tracker=FunctionGlobalTracker(self.fn, key),
        )
