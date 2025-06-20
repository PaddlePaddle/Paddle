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

import dataclasses
import operator
import sys
import types
from enum import Enum
from functools import cached_property, reduce
from typing import TYPE_CHECKING, Any

import numpy as np

import paddle
from paddle._typing import unreached
from paddle.framework import core
from paddle.jit.dy2static.utils import (
    dataclass_as_dict,
    dataclass_from_dict,
    is_dataclass_instance,
)
from paddle.jit.sot.opcode_translator.executor.pycode_generator import PyCodeGen
from paddle.pir.core import _PADDLE_PIR_DTYPE_2_NUMPY_DTYPE

from ....infer_meta import (
    DistInfo,
    MetaInfo,
    MetaInfoOrNull,
)
from ....symbolic.statement_ir import Symbol
from ....symbolic_shape.constraints import (
    AddConstraintNode,
    BitwiseAndConstraintNode,
    BitwiseLShiftConstraintNode,
    BitwiseNotConstraintNode,
    BitwiseOrConstraintNode,
    BitwiseRShiftConstraintNode,
    BitwiseXorConstraintNode,
    ConstantConstraintNode,
    ConstraintNode,
    EqualConstraintNode,
    FloorDivConstraintNode,
    GreaterEqualConstraintNode,
    GreaterThanConstraintNode,
    LessEqualConstraintNode,
    LessThanConstraintNode,
    LogicalNotConstraintNode,
    LogicalToBoolConstraintNode,
    ModConstraintNode,
    MulConstraintNode,
    NegativeConstraintNode,
    NotEqualConstraintNode,
    PowConstraintNode,
    SubConstraintNode,
    SymbolicConstraintNode,
    TrueDivConstraintNode,
)
from ....symbolic_shape.operators import (
    symbolic_not,
    symbolic_to_bool,
)
from ....symbolic_shape.symbolic_value import (
    SymbolicBool,
    SymbolicFloat,
    SymbolicInt,
    SymbolicValue,
)
from ....utils import (
    ENV_SOT_ALLOW_DYNAMIC_SHAPE,
    ENV_SOT_ENABLE_0_SIZE_FALLBACK,
    BreakGraphError,
    BuiltinFunctionBreak,
    ConditionalFallbackError,
    ConstTypes,
    DataDependencyDynamicShapeBreak,
    DataDependencyOperationBreak,
    FallbackError,
    NameGenerator,
    SotCapturedExceptionFactory,
    UnsupportedOperationBreak,
    get_tensor_methods,
    log,
    printable,
)
from ....utils.envs import ENV_SOT_BREAK_GRAPH_ON_GET_SYMBOLIC_VALUE
from ....utils.exceptions import (
    InnerError,
    UnsupportedPaddleAPIBreak,
)
from ..dispatch_functions import (
    place_get_device_id,
    place_get_device_type,
    tensor_dim,
)
from ..guard import (
    FasterStringifiedExpression,
    StringifiedExpression,
    check_faster_guard,
    check_guard,
    object_equal_faster_guard,
    object_equal_stringified_guard,
    stringify_pyobject,
    union_free_vars,
)
from ..mutable_data import MutableDictLikeData
from ..tracker import (
    ConstTracker,
    DanglingTracker,
    DummyTracker,
    GetAttrTracker,
    GetItemTracker,
    GetIterTracker,
    GlobalTracker,
    LocalTracker,
    SymbolicOperationTracker,
    Tracker,
)
from .base import VariableBase, VariableFactory

if TYPE_CHECKING:
    import numpy.typing as npt
    from typing_extensions import TypeAlias

    from ..function_graph import FunctionGraph
    from ..pycode_generator import PyCodeGen
    from .callable import (
        BuiltinVariable,
        ClassVariable,
        DataClassVariable,
        FunctionVariable,
    )
    from .container import TupleVariable

    SymbolicConstraint: TypeAlias = tuple[
        ConstraintNode, dict[str, "SymbolicVariable"]
    ]


FP_DTYPE_ABBRS = {
    core.DataType.BFLOAT16: "bfloat16",
    core.DataType.FLOAT64: "float64",
    core.DataType.FLOAT32: "float32",
    core.DataType.FLOAT16: "float16",
    core.DataType.FLOAT8_E4M3FN: "float8_e4m3fn",
    core.DataType.FLOAT8_E5M2: "float8_e5m2",
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

    def getitem(self, key):
        track_vars: list[VariableBase] = [self]
        if self.get_py_type() is not str:
            raise InnerError(
                f"getitem can only be applied to a str variable, but got {self.get_py_type()}"
            )
        if isinstance(key, VariableBase):
            track_vars.append(key)
            key = key.get_py_value()
        retval = self.value[key]
        return ConstantVariable(retval, self.graph, DummyTracker(track_vars))

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

    def flatten_inner_vars(self):
        return [
            inner_var
            for arg in list(self.args) + list(self.kwargs.values())
            for inner_var in arg.flatten_inner_vars()
        ]


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

    @check_faster_guard
    def make_faster_guard(self) -> list[paddle.framework.core.GuardNodeBase]:
        if isinstance(self.tracker, GetAttrTracker) and isinstance(
            self.tracker.obj, TensorVariable
        ):
            expr_node = self.tracker.obj.tracker.guard_tree_expr_node()
            assert paddle.framework.use_pir_api(), "Only support PIR"
            return [
                paddle.framework.core.GuardNode(
                    paddle.framework.core.DtypeMatchGuard(self.value),
                    [expr_node],
                )
            ]
        else:
            return object_equal_faster_guard(self)

    @check_guard
    def make_stringified_guard(self) -> list[StringifiedExpression]:
        if isinstance(self.tracker, GetAttrTracker) and isinstance(
            self.tracker.obj, TensorVariable
        ):
            tensor_value_tracer = (
                self.tracker.obj.tracker.trace_value_from_frame()
            )
            dtype_str, dtype_free_vars = stringify_pyobject(self.value)
            # TODO(cleanup-legacy-ir): Remove this branch after we remove legacy IR
            if not paddle.framework.use_pir_api():
                return [
                    StringifiedExpression(
                        f"MetaInfoOrNull.from_tensor({{}}).unwrap_unsafe().dtype == {dtype_str}",
                        [tensor_value_tracer],
                        union_free_vars(
                            {"MetaInfoOrNull": MetaInfoOrNull},
                            tensor_value_tracer.free_vars,
                            dtype_free_vars,
                        ),
                    )
                ]
            return [
                FasterStringifiedExpression(
                    f"{{}}.dtype == {dtype_str}",
                    paddle.framework.core.DtypeMatchGuard(self.value),
                    [tensor_value_tracer],
                    union_free_vars(
                        tensor_value_tracer.free_vars,
                        dtype_free_vars,
                    ),
                )
            ]

        else:
            return object_equal_stringified_guard(self)

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
        tensor (paddle.Tensor | MetaInfoOrNull): The tensor to be wrapped.
        graph (FunctionGraph): The FunctionGraph object that this variable is associated with.
        tracker (Tracker): The Tracker object that tracks the information of this variable.
    """

    var_name_generator = NameGenerator("var_")
    mutable_attrs = ["meta"]

    def __init__(
        self,
        meta: MetaInfoOrNull,
        graph: FunctionGraph,
        tracker: Tracker,
    ):
        super().__init__(graph, tracker)
        self.value = None
        self.meta = meta
        dynamic_axes: list[int] = []
        if (
            ENV_SOT_ALLOW_DYNAMIC_SHAPE.get()
            and self.tracker.is_traceable()
            and not self.meta.is_null()
        ):
            dynamic_axes = self.analyse_dynamic_axes(tracker)
        self.var_name = self.var_name_generator.next()
        self.graph.side_effects.record_mutable_variable(self)
        self.meta = self.meta.with_dynamic_axes(self.var_name, dynamic_axes)
        self.origin_meta = self.meta

    def analyse_dynamic_axes(self, tracker: Tracker):
        from ..executor_cache import OpcodeExecutorCache

        shape_dims = (
            self.shape.proxy.get_all()
        )  # Trigger convert all shape dims to Variable
        dynamic_axes = [
            i
            for i, dim in enumerate(shape_dims)
            if isinstance(dim, SymbolicVariable)
        ]
        if dynamic_axes:
            tracker_expr = tracker.trace_value_from_frame().inlined_expr
            symbolic_inputs = OpcodeExecutorCache().get_symbolic_inputs(
                self.graph.pycode_gen._origin_code
            )
            log(
                1,
                f"Start analyse dynamic axes for {tracker.trace_value_from_frame().inlined_expr} in {self.graph.pycode_gen._origin_code}\n",
            )
            for key, symbolic_input in symbolic_inputs.items():
                if key.startswith(tracker_expr):
                    log(1, f"  {key}: {symbolic_input}\n")
            log(
                1,
                f"  -> Tensor {tracker_expr} with dynamic axes {dynamic_axes}\n",
            )
        return dynamic_axes

    def __len__(self):
        if isinstance(self.meta.unwrap_or_breakgraph().shape[0], SymbolicInt):
            raise BreakGraphError(
                DataDependencyDynamicShapeBreak(
                    "length of tensor variable with first dimension is dynamic shape causes graph break."
                )
            )
        return self.meta.unwrap_or_breakgraph().shape[0]

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
            DataDependencyOperationBreak(
                "Called TensorVariable.get_py_value. Should not use Tensor's value in simulating."
            )
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

    @check_faster_guard
    def make_faster_guard(self) -> list[paddle.framework.core.GuardNodeBase]:
        assert paddle.framework.use_pir_api(), "Only support PIR"
        expr_node = self.tracker.guard_tree_expr_node()
        meta = self.origin_meta
        if meta.is_null():
            return [
                paddle.framework.core.GuardNode(
                    paddle.framework.core.IsNotDenseTensorHoldAllocationMatchGuard(),
                    [expr_node],
                ),
            ]
        meta = meta.unwrap_unsafe()
        return [
            # Check shape
            paddle.framework.core.GuardNode(
                paddle.framework.core.ShapeMatchGuard(meta.shape),
                [expr_node],
            ),
            # Check dtype
            paddle.framework.core.GuardNode(
                paddle.framework.core.DtypeMatchGuard(meta.dtype),
                [expr_node],
            ),
            # Check stop_gradient
            paddle.framework.core.GuardNode(
                paddle.framework.core.ValueMatchGuard(meta.stop_gradient),
                [
                    paddle.framework.core.AttributeExprNode(
                        expr_node, "stop_gradient"
                    )
                ],
            ),
            # Check dist info
            paddle.framework.core.TensorDistMetaMatchGuardNode(
                meta.dist_info,
                [
                    expr_node,
                    paddle.framework.core.ExternVarExprNode(
                        "___dist_info_from_tensor", DistInfo.from_tensor
                    ),
                ],
            ),
        ]

    @check_guard
    def make_stringified_guard(self) -> list[StringifiedExpression]:
        frame_value_tracer = self.tracker.trace_value_from_frame()

        # TODO(cleanup-legacy-ir): Remove this branch after we remove legacy IR
        if not paddle.framework.use_pir_api():
            if (
                ENV_SOT_ALLOW_DYNAMIC_SHAPE.get()
                and not self.origin_meta.is_null()
            ):
                str_left_expr = f"MetaInfoOrNull.from_tensor({{}}, dynamic_axes={self.meta.unwrap_unsafe().dynamic_axes}).guard_str()"
            else:
                str_left_expr = "MetaInfoOrNull.from_tensor({}).guard_str()"
            return [
                StringifiedExpression(
                    f"{str_left_expr} == '{self.origin_meta.guard_str()}'",
                    [frame_value_tracer],
                    union_free_vars(
                        {"MetaInfoOrNull": MetaInfoOrNull},
                        frame_value_tracer.free_vars,
                    ),
                )
            ]

        # A quick check path for PIR, we don't need dtype conversion for AMP in PIR
        meta = self.origin_meta
        if meta.is_null():
            return [
                StringifiedExpression(
                    "not {}._is_dense_tensor_hold_allocation()",
                    [frame_value_tracer],
                    union_free_vars(frame_value_tracer.free_vars),
                )
            ]
        meta = meta.unwrap_unsafe()
        dtype_str, dtype_free_vars = stringify_pyobject(meta.dtype)
        guards = [
            # Check rank
            StringifiedExpression(
                f"len({{}}.shape) == {len(meta.shape)}",
                [frame_value_tracer],
                union_free_vars(frame_value_tracer.free_vars),
            ),
            # Check each dim except dynamic dim
            *[
                (
                    StringifiedExpression(
                        f"{{}}.shape[{i}] == {meta.shape[i]}",
                        [frame_value_tracer],
                        union_free_vars(frame_value_tracer.free_vars),
                    )
                    if not isinstance(meta.shape[i], SymbolicInt)
                    else StringifiedExpression(
                        f"{{}}.shape[{i}] >= 1",
                        [frame_value_tracer],
                        union_free_vars(frame_value_tracer.free_vars),
                    )
                )
                for i in range(len(meta.shape))
            ],
            # Check dtype
            StringifiedExpression(
                f"{{}}.dtype == {dtype_str}",
                [frame_value_tracer],
                union_free_vars(frame_value_tracer.free_vars, dtype_free_vars),
            ),
            # Check stop_gradient
            StringifiedExpression(
                f"{{}}.stop_gradient == {meta.stop_gradient!r}",
                [frame_value_tracer],
                union_free_vars(frame_value_tracer.free_vars),
            ),
            # Check whether this tensor is distributed
            StringifiedExpression(
                f"{{}}.is_dist() is {(meta.dist_info is not None)!r}",
                [frame_value_tracer],
                union_free_vars(frame_value_tracer.free_vars),
            ),
        ]
        if meta.dist_info is not None:
            tensor_dist_info = meta.dist_info
            guards.extend(
                [
                    # check mesh shape
                    StringifiedExpression(
                        f"DistInfo.from_tensor({{}}).mesh.shape == {tensor_dist_info.mesh.shape}",
                        [frame_value_tracer],
                        union_free_vars(
                            frame_value_tracer.free_vars,
                            {
                                "paddle": paddle,
                                "DistInfo": DistInfo,
                            },
                        ),
                    ),
                    # check mesh process ids
                    StringifiedExpression(
                        f"DistInfo.from_tensor({{}}).mesh.process_ids == {tensor_dist_info.mesh.process_ids}",
                        [frame_value_tracer],
                        union_free_vars(
                            frame_value_tracer.free_vars,
                            {
                                "paddle": paddle,
                                "DistInfo": DistInfo,
                            },
                        ),
                    ),
                    # check dims mapping
                    StringifiedExpression(
                        f"DistInfo.from_tensor({{}}).dims_mapping == {tensor_dist_info.dims_mapping}",
                        [frame_value_tracer],
                        union_free_vars(
                            frame_value_tracer.free_vars,
                            {
                                "paddle": paddle,
                                "DistInfo": DistInfo,
                            },
                        ),
                    ),
                    # check local shape
                    StringifiedExpression(
                        f"DistInfo.from_tensor({{}}).local_shape == {tensor_dist_info.local_shape}",
                        [frame_value_tracer],
                        union_free_vars(
                            frame_value_tracer.free_vars,
                            {
                                "paddle": paddle,
                                "DistInfo": DistInfo,
                            },
                        ),
                    ),
                ]
            )
        return guards

    def get_iter(self):
        from .iter import SequenceIterVariable

        return SequenceIterVariable(self, self.graph, GetIterTracker(self))

    @property
    def main_info(self) -> dict[str, Any]:
        if self.meta.is_null():
            return {
                "meta": "null",
            }
        meta = self.meta.unwrap_unsafe()
        dtype = meta.dtype
        if isinstance(dtype, paddle.core.VarDesc.VarType):
            dtype = paddle.pir.core.vartype_to_datatype[dtype]
        return {
            "shape": meta.shape,
            "dtype": DTYPE_ABBRS[dtype],
            "stop_gradient": meta.stop_gradient,
            "var_name": self.var_name,
            "dist_info": meta.dist_info,
        }

    def getitem(self, key):
        return self.graph.call_tensor_method("__getitem__", self, key)

    def setitem(self, key, value):
        new_tensor = self.graph.call_paddle_api(
            paddle.static.setitem,
            self,
            key,
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

        perm = list(
            range(len(self.meta.unwrap_or_breakgraph().shape) - 1, -1, -1)
        )
        perm_var = ListVariable(perm, self.graph, tracker=ConstTracker(perm))
        out = self.graph.call_paddle_api(paddle.transpose, self, perm_var)
        return out

    @tensor_property
    def mT(self):
        """
        Return a new TensorVariable object that wraps the result of calling the mT method on the wrapped value of this TensorVariable.
        """
        from .container import ListVariable

        if len(self.meta.unwrap_or_breakgraph().shape) < 2:
            raise ValueError(
                f"Variable.ndim({self.ndim}) is required to be greater than or equal to 2."
            )

        perm = list(range(len(self.meta.unwrap_or_breakgraph().shape)))
        perm[-1], perm[-2] = perm[-2], perm[-1]
        perm_var = ListVariable(perm, self.graph, tracker=DummyTracker([self]))
        out = self.graph.call_paddle_api(paddle.transpose, self, perm_var)
        return out

    @tensor_property
    def ndim(self):
        """
        Return a ConstantVariable object that represents the number of dimensions of the wrapped value of this TensorVariable.
        """
        return ConstantVariable(
            len(self.meta.unwrap_or_breakgraph().shape),
            self.graph,
            DummyTracker([self]),
        )

    @tensor_property
    def size(self):
        """
        Return a ConstantVariable object that represents the total number of elements in the wrapped value of this TensorVariable.
        """
        from .callable import BuiltinVariable

        meta = self.meta.unwrap_or_breakgraph()

        if not meta.is_dynamic_shape():
            n_elements = reduce(operator.mul, meta.shape, 1)
            return ConstantVariable(
                n_elements, self.graph, DummyTracker([self])
            )
        if not ENV_SOT_ALLOW_DYNAMIC_SHAPE.get():
            raise BreakGraphError(
                DataDependencyDynamicShapeBreak(
                    f"Getting size for a dynamic shape tensor causes graph break. shape = {meta.shape}"
                )
            )
        return reduce(
            BuiltinVariable(operator.mul, self.graph, DanglingTracker()),
            self.shape,
            ConstantVariable.wrap_literal(1, self.graph),
        )

    @tensor_property
    def shape(self):
        meta = self.meta.unwrap_or_breakgraph()
        if not ENV_SOT_ALLOW_DYNAMIC_SHAPE.get() and meta.is_dynamic_shape():
            raise BreakGraphError(
                DataDependencyDynamicShapeBreak(
                    f"Getting shape for a dynamic shape tensor causes graph break. shape = {meta.shape}"
                )
            )
        from .container import ListVariable

        tracker = GetAttrTracker(self, "shape")
        return ListVariable(meta.shape, self.graph, tracker=tracker)

    def len(self):
        meta = self.meta.unwrap_or_breakgraph()
        if len(meta.shape) == 0:
            raise InnerError("len() of a 0-D tensor is wrong")
        first_dim = meta.shape[0]
        if not isinstance(first_dim, SymbolicInt):
            return ConstantVariable(first_dim, self.graph, DummyTracker([self]))
        if not ENV_SOT_ALLOW_DYNAMIC_SHAPE.get():
            raise BreakGraphError(
                DataDependencyDynamicShapeBreak(
                    "Getting len() for a dynamic shape tensor causes graph break."
                )
            )
        return self.shape[0]

    def is_tensor(self):
        return ConstantVariable(True, self.graph, DummyTracker([self]))

    def is_complex(self):
        dtype = self.meta.unwrap_or_breakgraph().dtype
        if isinstance(dtype, paddle.core.VarDesc.VarType):
            dtype = paddle.pir.core.vartype_to_datatype[dtype]
        is_cp_dtype = dtype in CP_DTYPE_ABBRS
        return ConstantVariable(is_cp_dtype, self.graph, DummyTracker([self]))

    def is_integer(self):
        dtype = self.meta.unwrap_or_breakgraph().dtype
        if isinstance(dtype, paddle.core.VarDesc.VarType):
            dtype = paddle.pir.core.vartype_to_datatype[dtype]
        is_int_dtype = dtype in INT_DTYPE_ABBRS
        return ConstantVariable(is_int_dtype, self.graph, DummyTracker([self]))

    def is_floating_point(self):
        dtype = self.meta.unwrap_or_breakgraph().dtype
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
            "dim": tensor_dim,
            "ndimension": tensor_dim,
            "is_tensor": paddle.is_tensor,
            "is_complex": paddle.is_complex,
            "is_integer": paddle.is_integer,
            "is_floating_point": paddle.is_floating_point,
        }
        if (
            name in ["name", "place", "type"]
            and self.meta.unwrap_or_breakgraph().is_inner_var()
        ):
            raise BreakGraphError(
                DataDependencyOperationBreak(
                    f"{self.meta.unwrap_or_breakgraph().name} is a middle tensor. Not support to get {name} property."
                )
            )
        if name in [
            "dtype",
            "type",
            "name",
            "persistable",
            "stop_gradient",
            "place",
        ]:
            return VariableFactory.from_value(
                getattr(self.meta.unwrap_or_breakgraph(), name),
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
            ).bind_dangling_fn(self, name)
        elif name in get_tensor_methods():
            from .callable import TensorFunctionVariable

            fn_var = TensorFunctionVariable(
                name, graph=self.graph, tracker=DanglingTracker()
            )
            return fn_var.bind_dangling_fn(self, name)
        else:
            raise BreakGraphError(
                UnsupportedPaddleAPIBreak(fn_name=f"Tensor.{name}")
            )

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
        raise BreakGraphError(
            BuiltinFunctionBreak("Don't support TensorVariable delattr")
        )

    @VariableFactory.register_from_value()
    def from_value(value: Any, graph: FunctionGraph, tracker: Tracker):
        if isinstance(value, (paddle.Tensor, MetaInfoOrNull)):
            value = (
                MetaInfoOrNull.from_tensor(value)
                if isinstance(value, paddle.Tensor)
                else value
            )
            return TensorVariable(value, graph, tracker)
        return None


def get_symbolic_from_meta(meta_or_null: MetaInfoOrNull) -> SymbolicValue:
    if meta_or_null.is_null():
        raise InnerError(
            f"get_symbolic_from_meta() got {meta_or_null}. Only MetaInfo is supported."
        )
    meta = meta_or_null.unwrap_unsafe()
    if meta.dtype in [paddle.bool]:
        value = SymbolicBool()
    elif meta.dtype in [
        paddle.int8,
        paddle.uint8,
        paddle.int16,
        paddle.int32,
        paddle.int64,
    ]:
        value = SymbolicInt()
    elif meta.dtype in [
        paddle.bfloat16,
        paddle.float16,
        paddle.float32,
        paddle.float64,
        paddle.float8_e4m3fn,
        paddle.float8_e5m2,
    ]:
        value = SymbolicFloat()
    else:
        raise InnerError(f"Unsupported dtype {meta.dtype} for SymbolicVariable")
    return value


class SymbolicVariable(VariableBase):
    """
    SymbolicVariable is a subclass of VariableBase used to wrap a symbolic value.

    Args:
        value_or_meta (SymbolicInt | MetaInfoOrNull): The symbolic value  to be wrapped or metadata.
        graph (FunctionGraph): The FunctionGraph object that this variable is associated with.
        tracker (Tracker): The Tracker object that tracks the information of this variable.
    """

    var_name_generator = NameGenerator("symint_")
    value: SymbolicValue
    mutable_attrs = ["need_guard_value", "constraints"]

    def __init__(
        self,
        value_or_meta: SymbolicInt | MetaInfoOrNull,
        graph: FunctionGraph,
        tracker: Tracker,
    ):
        super().__init__(graph, tracker)
        self.var_name = self.var_name_generator.next()
        if isinstance(value_or_meta, MetaInfoOrNull):
            assert (
                not value_or_meta.is_null()
            ), "MetaInfoOrNull should not be null"
            assert len(value_or_meta.unwrap_unsafe().shape) == 0
            self.value = get_symbolic_from_meta(value_or_meta)
            self.meta = value_or_meta
        else:
            assert isinstance(
                value_or_meta, SymbolicInt
            ), f"Unsupported type {type(value_or_meta)} for SymbolicVariable"
            self.value = value_or_meta
            self.meta = MetaInfo(
                [], paddle.int64, True, self.var_name, False, None, None
            ).wrap()
        self.need_guard_value = False
        self.graph.side_effects.record_mutable_variable(self)
        self.constraints: list[SymbolicConstraint] = []
        if self.value.is_backed():
            # The inherent constraint of the symbolic variable is that it must be greater than or equal to 2.
            self.add_constraint(
                (
                    GreaterEqualConstraintNode(
                        SymbolicConstraintNode(self.var_name),
                        ConstantConstraintNode(1),
                    ),
                    {self.var_name: self},
                )
            )

    def add_constraint(self, constraint: SymbolicConstraint):
        constraint_node, constraint_extern_vars = constraint
        for extern_var in constraint_extern_vars.values():
            assert isinstance(
                extern_var, SymbolicVariable
            ), f"SymbolicVariable.add_constraint() got {extern_var}."
            assert (
                extern_var.value.is_backed()
            ), "Only backed symbol is supported."
            assert (
                extern_var.tracker.is_traceable()
            ), "Only traceable symbol is supported."
        self.constraints.append(constraint)

    def to_constant(self):
        from ..executor_cache import (
            OpcodeExecutorCache,
        )

        symbolic_inputs = OpcodeExecutorCache().get_symbolic_inputs(
            self.graph.pycode_gen._origin_code
        )

        disabled_vars = set()

        def disable_symbolic(var: VariableBase):
            if var in disabled_vars:
                return

            disabled_vars.add(var)
            if var.tracker.is_traceable():
                tracker_expr = var.tracker.trace_value_from_frame().inlined_expr
                symbolic_inputs[tracker_expr] = None
                return
            for input_var in var.tracker.inputs:
                disable_symbolic(input_var)

        disable_symbolic(self)
        self.graph.need_cache = False
        log(3, f"Fallback {self} to ConstantVariable\n")
        return ConstantVariable(
            self.get_py_value(), self.graph, DummyTracker([self])
        )

    def get_py_value(self, allow_tensor: bool = False) -> bool | int | float:
        if ENV_SOT_BREAK_GRAPH_ON_GET_SYMBOLIC_VALUE.get():
            raise BreakGraphError(
                DataDependencyOperationBreak(
                    "get_py_value from SymbolicVariable"
                )
            )
        self.need_guard_value = True
        log(
            3,
            f"get_py_value from SymbolicVariable {self} caused value need guard\n",
        )
        if self.value.is_backed():
            return self.value.get_example_value()
        inputs = self.tracker.inputs
        input_values = [x.get_py_value() for x in inputs]
        if not isinstance(self.tracker, SymbolicOperationTracker):
            raise BreakGraphError(
                DataDependencyOperationBreak(
                    f"SymbolicVariable.get_py_value() got {self.tracker}. May be a symbol from a inner tensor."
                )
            )
        value = self.tracker.op(*input_values)
        assert isinstance(
            value, (bool, int, float)
        ), f"SymbolicVariable.get_py_value() should return bool, int or float, but got {type(value)}"
        return value

    def get_example_value(
        self, allow_tensor: bool = False
    ) -> bool | int | float:
        if self.value.is_backed():
            return self.value.get_example_value()
        inputs = self.tracker.inputs
        input_values = []
        for input in inputs:
            if isinstance(input, ConstantVariable):
                input_values.append(input.get_py_value())
            elif isinstance(input, SymbolicVariable):
                input_values.append(input.get_example_value())
            else:
                raise BreakGraphError(
                    DataDependencyOperationBreak(
                        f"SymbolicVariable.get_example_value() got {input}. May be a symbol from a inner tensor."
                    )
                )
        if not isinstance(self.tracker, SymbolicOperationTracker):
            raise BreakGraphError(
                DataDependencyOperationBreak(
                    f"SymbolicVariable.get_example_value() got {self.tracker}. May be a symbol from a inner tensor."
                )
            )
        value = self.tracker.op(*input_values)
        assert isinstance(
            value, (bool, int, float)
        ), f"SymbolicVariable.get_example_value() should return bool, int or float, but got {type(value)}"
        return value

    def create_constraint_tree(
        self,
    ) -> tuple[ConstraintNode, dict[str, SymbolicVariable]]:
        tracker = self.tracker
        if not isinstance(tracker, SymbolicOperationTracker):
            return SymbolicConstraintNode(self.var_name), {self.var_name: self}
        input_nodes = []
        extern_vars = {}
        num_sym = 0
        for input in tracker.inputs:
            assert isinstance(
                input, (ConstantVariable, SymbolicVariable)
            ), f"SymbolicVariable.create_constraint_tree() got {input}."
            if isinstance(input, ConstantVariable):
                input_nodes.append(ConstantConstraintNode(input.get_py_value()))
            else:
                num_sym += 1
                input_node, input_extern_vars = input.create_constraint_tree()
                input_nodes.append(input_node)
                extern_vars.update(input_extern_vars)

        # TODO(SigureMo): use a better way to dispatch constraint node
        # Arithmetic operations
        if tracker.op is operator.neg:
            assert len(input_nodes) == 1
            return NegativeConstraintNode(input_nodes[0]), extern_vars
        elif tracker.op is operator.invert:
            assert len(input_nodes) == 1
            return BitwiseNotConstraintNode(input_nodes[0]), extern_vars
        elif tracker.op is operator.add:
            assert len(input_nodes) == 2
            return AddConstraintNode(*input_nodes), extern_vars
        elif tracker.op is operator.sub:
            assert len(input_nodes) == 2
            return SubConstraintNode(*input_nodes), extern_vars
        elif tracker.op is operator.mul:
            assert len(input_nodes) == 2
            return MulConstraintNode(*input_nodes), extern_vars
        elif tracker.op is operator.truediv:
            assert len(input_nodes) == 2
            return TrueDivConstraintNode(*input_nodes), extern_vars
        elif tracker.op is operator.floordiv:
            assert len(input_nodes) == 2
            return FloorDivConstraintNode(*input_nodes), extern_vars
        elif tracker.op is operator.mod:
            assert len(input_nodes) == 2
            return ModConstraintNode(*input_nodes), extern_vars
        elif tracker.op is operator.pow:
            assert len(input_nodes) == 2
            return PowConstraintNode(*input_nodes), extern_vars
        elif tracker.op is operator.and_:
            assert len(input_nodes) == 2
            return BitwiseAndConstraintNode(*input_nodes), extern_vars
        elif tracker.op is operator.lshift:
            assert len(input_nodes) == 2
            return BitwiseLShiftConstraintNode(*input_nodes), extern_vars
        elif tracker.op is operator.rshift:
            assert len(input_nodes) == 2
            return BitwiseRShiftConstraintNode(*input_nodes), extern_vars
        elif tracker.op is operator.or_:
            assert len(input_nodes) == 2
            return BitwiseOrConstraintNode(*input_nodes), extern_vars
        elif tracker.op is operator.xor:
            assert len(input_nodes) == 2
            return BitwiseXorConstraintNode(*input_nodes), extern_vars
        # Logical operations
        elif tracker.op is symbolic_to_bool:
            assert len(input_nodes) == 1
            return (
                LogicalToBoolConstraintNode(input_nodes[0]),
                extern_vars,
            )
        elif tracker.op is symbolic_not:
            assert len(input_nodes) == 1
            return LogicalNotConstraintNode(input_nodes[0]), extern_vars
        elif tracker.op is operator.eq:
            assert len(input_nodes) == 2
            return EqualConstraintNode(*input_nodes), extern_vars
        elif tracker.op is operator.ne:
            assert len(input_nodes) == 2
            return NotEqualConstraintNode(*input_nodes), extern_vars
        elif tracker.op is operator.le:
            assert len(input_nodes) == 2
            return LessEqualConstraintNode(*input_nodes), extern_vars
        elif tracker.op is operator.lt:
            assert len(input_nodes) == 2
            return LessThanConstraintNode(*input_nodes), extern_vars
        elif tracker.op is operator.ge:
            assert len(input_nodes) == 2
            return GreaterEqualConstraintNode(*input_nodes), extern_vars
        elif tracker.op is operator.gt:
            assert len(input_nodes) == 2
            return GreaterThanConstraintNode(*input_nodes), extern_vars
        else:
            raise InnerError(f"Unsupported symbolic operation {tracker.op}.")

    def get_py_type(self):
        if isinstance(self.value, int):
            return int
        return self.value.get_static_type()

    def get_symbol(self) -> Symbol:
        return Symbol(self.var_name)

    def __bool__(self) -> bool:
        return bool(self.get_py_value())

    def bool(self):
        return ConstantVariable(bool(self), self.graph, DummyTracker([self]))

    def __int__(self) -> int:
        return int(self.get_py_value())

    def int(self):
        return ConstantVariable(int(self), self.graph, DummyTracker([self]))

    def __float__(self) -> float:
        return float(self.get_py_value())

    def float(self):
        return ConstantVariable(float(self), self.graph, DummyTracker([self]))

    def __complex__(self) -> complex:
        return complex(self.get_py_value())

    def complex(self):
        return ConstantVariable(complex(self), self.graph, DummyTracker([self]))

    @property
    def out_var_name(self):
        return f"{self.graph.OUT_VAR_PREFIX}{self.var_name}"

    def _reconstruct(self, codegen: PyCodeGen):
        codegen.gen_load_fast(self.out_var_name)
        codegen.gen_load_method("item")
        codegen.gen_call_method(0)

    @check_faster_guard
    def make_faster_guard(self) -> list[paddle.framework.core.GuardNodeBase]:
        assert ENV_SOT_ALLOW_DYNAMIC_SHAPE.get()
        from ..executor_cache import OpcodeExecutorCache

        expr_node = self.tracker.guard_tree_expr_node()
        frame_value_tracer = self.tracker.trace_value_from_frame()
        # TODO(zrr1999): symbolic_inputs need frame_value_tracer.inlined_expr
        symbolic_inputs = OpcodeExecutorCache().get_symbolic_inputs(
            self.graph.pycode_gen._origin_code
        )
        assert frame_value_tracer.inlined_expr in symbolic_inputs

        if self.need_guard_value:
            log(3, f"Need guard value for {self} in {expr_node}\n")
            return super().make_faster_guard()
        constraint_guards: list[paddle.framework.core.GuardNodeBase] = []
        for constraint in self.constraints:
            constraint_node, constraint_extern_vars = constraint
            extern_vars = {
                var_name: var.tracker.guard_tree_expr_node()
                for var_name, var in constraint_extern_vars.items()
            }
            constraint_guards.append(
                paddle.framework.core.ExprGuardNode(
                    constraint_node.create_guard_node(extern_vars)
                )
            )
        guards = [
            paddle.framework.core.GuardNode(
                paddle.core.TypeMatchGuard(self.get_py_type()),
                [expr_node],
            ),
            *constraint_guards,
        ]
        return guards

    @check_guard
    def make_stringified_guard(self) -> list[StringifiedExpression]:
        assert ENV_SOT_ALLOW_DYNAMIC_SHAPE.get()
        from ..executor_cache import OpcodeExecutorCache

        # NOTE(zrr1999): SymbolicVariable is not supported in faster guard mode

        frame_value_tracer = self.tracker.trace_value_from_frame()
        symbolic_inputs = OpcodeExecutorCache().get_symbolic_inputs(
            self.graph.pycode_gen._origin_code
        )

        assert frame_value_tracer.inlined_expr in symbolic_inputs

        if self.need_guard_value:
            log(3, f"Need guard value for {self} in {frame_value_tracer}\n")
            return super().make_stringified_guard()
        constraint_guards = []
        for constraint in self.constraints:
            constraint_node, constraint_extern_vars = constraint
            extern_vars = {
                var_name: var.tracker.trace_value_from_frame()
                for var_name, var in constraint_extern_vars.items()
            }
            constraint_guards.append(
                constraint_node.create_guard_expr(extern_vars)
            )
        guards = [
            FasterStringifiedExpression(
                f"id(type({{}})) == {id(self.get_py_type())}",
                paddle.core.TypeMatchGuard(self.get_py_type()),
                [frame_value_tracer],
                union_free_vars(frame_value_tracer.free_vars),
            ),
            *constraint_guards,
        ]
        return guards

    @staticmethod
    def should_create_symbolic_variable(
        value: Any,
        tracker: Tracker,
        symbolic_inputs: dict[str, dict[int, int] | None],
    ) -> bool | None:
        tracker_expr = tracker.trace_value_from_frame().inlined_expr
        symbolic_inputs.setdefault(tracker_expr, {})
        if tracker_expr in symbolic_inputs:
            symbolic_input = symbolic_inputs[tracker_expr]
            if symbolic_input is None:
                return False
            symbolic_input.setdefault(value, 0)
            symbolic_input[value] += 1
            if value == 0:
                return None  # Fallback to dygraph
            if value < 1:  # Specialize 0 and 1
                return False
            if len(symbolic_input.keys()) > 1:
                return True
            return False
        return False

    @staticmethod
    def find_tensor_shape_source(dim_tracker: Tracker):
        from .container import ListVariable

        if not isinstance(dim_tracker, GetItemTracker):
            return None
        if not isinstance(dim_tracker.container, ListVariable):
            return None
        if not isinstance(dim_tracker.container.tracker, GetAttrTracker):
            return None
        if dim_tracker.container.tracker.attr != "shape":
            return None
        if not isinstance(dim_tracker.container.tracker.obj, TensorVariable):
            return None
        tensor_var = dim_tracker.container.tracker.obj
        shape_idx = dim_tracker.key
        return tensor_var, shape_idx

    @staticmethod
    def has_local_leaf(tracker: Tracker) -> bool:
        if isinstance(tracker, LocalTracker):
            return True
        return any(
            SymbolicVariable.has_local_leaf(input.tracker)
            for input in tracker.inputs
        )

    @VariableFactory.register_from_value(successor="ConstantVariable")
    def from_value(value: Any, graph: FunctionGraph, tracker: Tracker):
        if not ENV_SOT_ALLOW_DYNAMIC_SHAPE.get():
            return None
        if isinstance(value, SymbolicInt):
            tensor_shape_source_result = (
                SymbolicVariable.find_tensor_shape_source(tracker)
            )
            assert tensor_shape_source_result is not None
            tensor_var, shape_idx = tensor_shape_source_result
            tensor_call_shape_var = graph.call_paddle_api(
                paddle.shape, tensor_var
            )
            tensor_dim_var = graph.call_tensor_method(
                "__getitem__",
                tensor_call_shape_var,
                ConstantVariable.wrap_literal(shape_idx, graph),
            )
            sym_var_tracker = (
                tracker
                if tensor_var.tracker.is_traceable()
                else DummyTracker([tensor_dim_var])
            )
            sym_var = SymbolicVariable(value, graph, sym_var_tracker)
            graph.add_alias(tensor_dim_var, sym_var)
            return sym_var
        if type(value) is not int:
            return None
        if not tracker.is_traceable():
            return None
        if not SymbolicVariable.has_local_leaf(tracker):
            return None

        from ..executor_cache import OpcodeExecutorCache

        symbolic_inputs = OpcodeExecutorCache().get_symbolic_inputs(
            graph.pycode_gen._origin_code
        )

        should_create_sym = SymbolicVariable.should_create_symbolic_variable(
            value, tracker, symbolic_inputs
        )
        if (
            should_create_sym is None
            and ENV_SOT_ENABLE_0_SIZE_FALLBACK.get()
            and SymbolicVariable.find_tensor_shape_source(tracker) is not None
        ):
            graph.add_global_guarded_variable(
                ConstantVariable(value, graph, tracker)
            )
            raise ConditionalFallbackError(
                "Fallback graph since input has 0 size Tensor",
                disable_eval_frame=True,
            )
        elif should_create_sym:
            return SymbolicVariable(SymbolicInt(value), graph, tracker)
        return None


class ParameterVariable(TensorVariable):
    var_name_generator = NameGenerator("param_")

    def __init__(
        self,
        meta: MetaInfoOrNull,
        graph: FunctionGraph,
        tracker: Tracker,
    ):
        super().__init__(meta, graph, tracker)

    @VariableFactory.register_from_value(successor="TensorVariable")
    def from_value(value: Any, graph: FunctionGraph, tracker: Tracker):
        if isinstance(value, (paddle.base.framework.EagerParamBase)):
            value = MetaInfoOrNull.from_tensor(value)
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

    make_stringified_guard = object_equal_stringified_guard
    make_faster_guard = object_equal_faster_guard

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


class SuperVariable(VariableBase):
    """
    Enhanced support for `super()` calls in Python.
    The `super()` function facilitates method delegation to parent classes
    following the method resolution order (MRO).

    Args:
        obj(Any): The object to be wrapped.
        graph(FunctionGraph): The FunctionGraph object that this variable is associated with.
        tracker(Tracker): The Tracker object that tracks the information of this variable.
    """

    def __init__(self, cls: ClassVariable, obj: VariableBase, graph, tracker):
        super().__init__(graph, tracker)
        self.cls = cls
        self.obj = obj

    @property
    def main_info(self) -> dict[str, Any]:
        if printable(self.obj):
            return {
                "value": f"super({self.cls.get_py_value().__name__}, {self.obj})"
            }
        return {"value": f"super({self.cls.get_py_value().__name__}, self)"}

    def get_py_value(self, allow_tensor=False) -> Any:
        cls = self.cls.get_py_value()
        obj = self.obj.get_py_value()
        return super(cls, obj)

    @check_guard
    def make_stringified_guard(self) -> list[StringifiedExpression]:
        guards = []
        if self.cls.tracker.need_guard():
            guards.extend(self.cls.make_stringified_guard())
        if self.obj.tracker.need_guard():
            guards.extend(self.obj.make_stringified_guard())
        return guards

    def getattr(self, name: str, default=None) -> VariableBase:
        from .callable import FunctionVariable

        mro = VariableFactory.from_value(
            self.cls.get_py_value().__mro__,
            self.graph,
            GetAttrTracker(self.cls, "__mro__"),
        )
        # `__mro__` contains currently class, so remove it
        super_mro = mro.get_wrapped_items()[1:]

        for super_cls in super_mro:
            if not super_cls.hasattr(name):
                continue
            attr = super_cls.getattr(name)
            if isinstance(attr, FunctionVariable):
                attr = attr.bind(self.obj, name)
            return attr

        return super().getattr(name)

    @VariableFactory.register_from_value()
    def from_value(value: Any, graph: FunctionGraph, tracker: Tracker):
        if not isinstance(value, super):
            return None
        cls = VariableFactory.from_value(
            value.__thisclass__, graph, DanglingTracker()
        )
        obj = VariableFactory.from_value(
            value.__self__, graph, DanglingTracker()
        )
        super_var = SuperVariable(cls, obj, graph, tracker)
        cls.tracker = GetAttrTracker(super_var, "__thisclass__")
        obj.tracker = GetAttrTracker(super_var, "__self__")
        return super_var


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

    @check_faster_guard
    def make_faster_guard(self) -> list[paddle.framework.core.GuardNodeBase]:
        expr_node = self.tracker.guard_tree_expr_node()
        return [
            paddle.framework.core.GuardNode(
                paddle.framework.core.TypeMatchGuard(slice),
                [expr_node],
            ),
            *self.getattr("start").make_faster_guard(),
            *self.getattr("stop").make_faster_guard(),
            *self.getattr("step").make_faster_guard(),
        ]

    @check_guard
    def make_stringified_guard(self) -> list[StringifiedExpression]:
        frame_value_tracer = self.tracker.trace_value_from_frame()
        result = [
            FasterStringifiedExpression(
                f"id(type({{}})) == {id(slice)}",
                paddle.framework.core.TypeMatchGuard(slice),
                [frame_value_tracer],
                frame_value_tracer.free_vars,
            ),
            *self.getattr("start").make_stringified_guard(),
            *self.getattr("stop").make_stringified_guard(),
            *self.getattr("step").make_stringified_guard(),
        ]
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
        raise BreakGraphError(
            UnsupportedOperationBreak(
                reason_str="Don't support SliceVariable setattr"
            )
        )

    def delattr(self, key):
        raise BreakGraphError(
            UnsupportedOperationBreak(
                reason_str="Don't support SliceVariable delattr"
            )
        )

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
    make_stringified_guard = object_equal_stringified_guard
    make_faster_guard = object_equal_faster_guard


class DygraphTracerVariable(VariableBase):
    # TODO(SigureMo): Remove this trick after we add CompareTracker
    def __init__(self, value, graph, tracker):
        super().__init__(graph, tracker)
        self.value = value

    def get_py_value(self, allow_tensor=False):
        return self.value

    @check_faster_guard
    def make_faster_guard(self) -> list[paddle.framework.core.GuardNodeBase]:
        return []

    @check_guard
    def make_stringified_guard(self) -> list[StringifiedExpression]:
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


class NumPyVariable(VariableBase):
    """
    NumPyVariable is a subclass of VariableBase used to wrap a Variable of the numpy type.

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

    @staticmethod
    def format_dtype(dtype: np.dtype):
        if (
            np.lib.NumpyVersion(np.__version__) >= "1.20.0"
            and dtype == np.bool_
        ):
            return "np.bool_"
        return f"np.{dtype}"

    @staticmethod
    def format_number(number: np.number):
        return f"{NumPyVariable.format_dtype(number.dtype)}({number.item()})"

    @check_faster_guard
    def make_faster_guard(self) -> list[paddle.framework.core.GuardNodeBase]:
        raise NotImplementedError(
            f"{self.__class__.__name__}.make_faster_guard is not implemented"
        )

    def make_stringified_guard(self) -> None:
        raise NotImplementedError


class NumPyNumberVariable(NumPyVariable):
    def _reconstruct(self, codegen: PyCodeGen):
        np_type = self.get_py_type()
        type_id = f"___np_{np_type.__name__}"
        codegen.gen_load_object(np_type, type_id)
        codegen.gen_load_const(self.value.item())
        codegen.gen_call_function(1)

    def getattr(self, name: str, default=None):
        from .callable import BuiltinVariable

        if name != "item":
            return super().getattr(name, default)
        return BuiltinVariable(
            np.number.item, self.graph, GetAttrTracker(self, name)
        ).bind(self, name)

    @check_faster_guard
    def make_faster_guard(self) -> list[paddle.framework.core.GuardNodeBase]:
        expr_node = self.tracker.guard_tree_expr_node()
        dtype_guard = paddle.framework.core.GuardNode(
            paddle.framework.core.NumPyDtypeMatchGuard(
                self.get_py_value().dtype
            ),
            [expr_node],
        )

        return [
            dtype_guard,
            paddle.framework.core.GuardNode(
                paddle.framework.core.ValueMatchGuard(self.get_py_value()),
                [expr_node],
            ),
        ]

    @check_guard
    def make_stringified_guard(self) -> list[StringifiedExpression]:
        frame_value_tracer = self.tracker.trace_value_from_frame()

        dtype_guard = FasterStringifiedExpression(
            f"{{}}.dtype == {NumPyVariable.format_dtype(self.get_py_value().dtype)}",
            paddle.framework.core.NumPyDtypeMatchGuard(
                self.get_py_value().dtype
            ),
            [frame_value_tracer],
            union_free_vars(frame_value_tracer.free_vars, {"np": np}),
        )

        return [
            dtype_guard,
            FasterStringifiedExpression(
                f"{{}} == {NumPyVariable.format_number(self.get_py_value())}",
                paddle.framework.core.ValueMatchGuard(self.get_py_value()),
                [frame_value_tracer],
                union_free_vars(frame_value_tracer.free_vars, {"np": np}),
            ),
        ]

    @VariableFactory.register_from_value()
    def from_value(value: Any, graph: FunctionGraph, tracker: Tracker):
        if isinstance(value, np.number):
            return NumPyNumberVariable(value, graph, tracker)
        return None


class NumPyBoolVariable(NumPyNumberVariable):
    @VariableFactory.register_from_value()
    def from_value(value: Any, graph: FunctionGraph, tracker: Tracker):
        if isinstance(value, np.bool_):
            return NumPyBoolVariable(value, graph, tracker)
        return None


class NumPyArrayVariable(NumPyVariable):
    var_name_generator = NameGenerator("np_var_")
    value: npt.NDArray[Any]
    mutable_attrs: list[str] = ["meta"]

    def __init__(
        self,
        value_or_meta: npt.NDArray[Any] | MetaInfoOrNull,
        graph: FunctionGraph,
        tracker: Tracker,
    ):
        super().__init__(None, graph, tracker)
        self.var_name = self.var_name_generator.next()
        self.graph.side_effects.record_mutable_variable(self)

        if isinstance(value_or_meta, MetaInfoOrNull):
            # TODO(wangmingkai02): self.value
            self.value = None
            self.meta = value_or_meta
        else:
            self.value = value_or_meta
            self.meta = MetaInfoOrNull.from_numpy(self.value)

    def __len__(self):
        meta = self.meta.unwrap_unsafe()
        if isinstance(meta.shape[0], SymbolicInt):
            raise BreakGraphError(
                DataDependencyDynamicShapeBreak(
                    "length of NumPy array Variable with first dimension is dynamic shape causes graph break."
                )
            )
        return meta.shape[0]

    def get_py_type(self):
        return np.ndarray

    def get_py_value(self, allow_tensor=False) -> Any:
        raise BreakGraphError(
            UnsupportedOperationBreak(
                reason_str="NumPyArrayVariable doesn't support get_py_value operation."
            )
        )

    def get_symbol(self) -> Symbol:
        return Symbol(self.var_name)

    def get_iter(self):
        from .iter import SequenceIterVariable

        return SequenceIterVariable(self, self.graph, GetIterTracker(self))

    @VariableFactory.register_from_value()
    def from_value(value: Any, graph: FunctionGraph, tracker: Tracker):
        if isinstance(value, np.ndarray):
            return NumPyArrayVariable(value, graph, tracker)
        return None

    @property
    def out_var_name(self):
        return f"{self.graph.OUT_VAR_PREFIX}{self.var_name}"

    def _reconstruct(self, codegen: PyCodeGen):
        codegen.gen_load_fast(self.out_var_name)
        codegen.gen_load_method("numpy")
        codegen.gen_call_method(0)

    @check_faster_guard
    def make_faster_guard(self) -> list[paddle.framework.core.GuardNodeBase]:
        meta = self.meta.unwrap_unsafe()
        expr_node = self.tracker.guard_tree_expr_node()
        type_guard = paddle.framework.core.GuardNode(
            paddle.framework.core.TypeMatchGuard(self.get_py_type()),
            [expr_node],
        )
        dtype_guard = paddle.framework.core.GuardNode(
            paddle.framework.core.NumPyDtypeMatchGuard(
                np.dtype(_PADDLE_PIR_DTYPE_2_NUMPY_DTYPE[meta.dtype])
            ),
            [expr_node],
        )
        shape_guard = paddle.framework.core.GuardNode(
            paddle.framework.core.NumPyArrayShapeMatchGuard(meta.shape),
            [expr_node],
        )
        return [type_guard, dtype_guard, shape_guard]

    @check_guard
    def make_stringified_guard(self) -> list[StringifiedExpression]:
        frame_value_tracer = self.tracker.trace_value_from_frame()
        meta = self.meta.unwrap_unsafe()

        dtype_guard = FasterStringifiedExpression(
            f"{{}}.dtype == {NumPyVariable.format_dtype(np.dtype(_PADDLE_PIR_DTYPE_2_NUMPY_DTYPE[meta.dtype]))}",
            paddle.framework.core.NumPyDtypeMatchGuard(
                np.dtype(_PADDLE_PIR_DTYPE_2_NUMPY_DTYPE[meta.dtype])
            ),
            [frame_value_tracer],
            union_free_vars(frame_value_tracer.free_vars, {"np": np}),
        )

        return [
            FasterStringifiedExpression(
                f"id(type({{}})) == {id(self.get_py_type())}",
                paddle.core.TypeMatchGuard(self.get_py_type()),
                [frame_value_tracer],
                union_free_vars(frame_value_tracer.free_vars),
            ),
            dtype_guard,
            StringifiedExpression(
                f"len({{}}.shape) == {len(meta.shape)}",
                [frame_value_tracer],
                union_free_vars(frame_value_tracer.free_vars),
            ),
            *[
                (
                    StringifiedExpression(
                        f"{{}}.shape[{i}] == {meta.shape[i]}",
                        [frame_value_tracer],
                        union_free_vars(frame_value_tracer.free_vars),
                    )
                    if not isinstance(meta.shape[i], SymbolicInt)
                    else StringifiedExpression(
                        f"{{}}.shape[{i}] >= 1",
                        [frame_value_tracer],
                        union_free_vars(frame_value_tracer.free_vars),
                    )
                )
                for i in range(len(meta.shape))
            ],
        ]


class PlaceVariable(ObjectVariable):
    def __init__(self, obj, graph, tracker):
        super().__init__(obj, graph, tracker)

    def getattr(self, name: str, default=None):
        if default is not None:
            raise FallbackError(
                "default argument for getattr is not implemented"
            )
        if name not in ["get_device_id", "get_device_type"]:
            return super().getattr(name, default)
        from .callable import BuiltinVariable

        if name == "get_device_id":
            return BuiltinVariable(
                place_get_device_id, self.graph, DanglingTracker()
            ).bind_dangling_fn(self, name)
        elif name == "get_device_type":
            return BuiltinVariable(
                place_get_device_type, self.graph, DanglingTracker()
            ).bind_dangling_fn(self, name)
        unreached()

    def get_device_id(self):
        return VariableFactory.from_value(
            self.value.get_device_id(), self.graph, DummyTracker([self])
        )

    def get_device_type(self):
        return VariableFactory.from_value(
            self.value.get_device_type(), self.graph, DummyTracker([self])
        )

    @VariableFactory.register_from_value()
    def from_value(value: Any, graph: FunctionGraph, tracker: Tracker):
        if paddle.is_compiled_with_cuda() and isinstance(
            value, (paddle.CUDAPlace, paddle.CUDAPinnedPlace)
        ):
            return PlaceVariable(value, graph, tracker)
        if paddle.is_compiled_with_xpu() and isinstance(
            value, (paddle.XPUPlace, paddle.XPUPinnedPlace)
        ):
            return PlaceVariable(value, graph, tracker)
        if isinstance(value, paddle.CustomPlace):
            return PlaceVariable(value, graph, tracker)
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
        if sys.version_info >= (3, 13):
            codegen.gen_push_null()
            return
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
                f"[{self.__class__.__name__}] received {key} to get value."
            )
        return self.proxy.get(key)

    def set(self, key, value):
        if isinstance(key, VariableBase):
            raise InnerError(
                f"[{self.__class__.__name__}] received {key} as key."
            )
        if not isinstance(value, VariableBase):
            raise InnerError(
                f"[{self.__class__.__name__}] received {value} to set value."
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
        from ..tracker import FunctionGlobalTracker

        if key not in proxy.original_data:
            return MutableDictLikeData.Empty()
        return VariableFactory.from_value(
            proxy.original_data[key],
            self.graph,
            tracker=FunctionGlobalTracker(self.fn, key),
        )


class ExceptionVariable(VariableBase):
    # The ExceptionVariable corresponds to the Exception class in Python
    mutable_attrs = [
        "__context__",
        "__cause__",
        "__suppress_context__",
        "__traceback__",
    ]

    def __init__(
        self,
        exc: Exception,
        graph: FunctionGraph = None,
        tracker: Tracker = None,
    ) -> None:
        super().__init__(graph=graph, tracker=tracker)

        self.exc = exc
        self.exc_type = exc.__class__
        self.args_variables = VariableFactory.from_value(
            exc.args, graph=graph, tracker=GetAttrTracker(self, "args")
        )

        self.__context__ = VariableFactory.from_value(
            self.exc.__context__,
            graph=graph,
            tracker=GetAttrTracker(self, "__context__"),
        )

        # raise ... from ...
        self.__cause__ = VariableFactory.from_value(
            self.exc.__cause__,
            graph=graph,
            tracker=GetAttrTracker(self, "__cause__"),
        )

        self.__suppress_context__ = VariableFactory.from_value(
            self.exc.__suppress_context__,
            graph=graph,
            tracker=GetAttrTracker(self, "__suppress_context__"),
        )

        # NOTE: Currently, since our primary goal is to trace the network structure of variables,
        # __traceback__ is always set to None.
        self.__traceback__ = ConstantVariable.wrap_literal(None, self.graph)

        self.graph.side_effects.record_mutable_variable(self)

    def get_py_type(self):
        return self.exc_type

    def get_py_value(self):
        return self.exc

    @property
    def main_info(self) -> dict[str, Any]:
        return {
            "exception": self.exc,
        }

    def setattr(self, key: str, value):
        # TODO(DrRyanHuang): Add UserDefinedException to __context__ and __cause__
        # TODO(DrRyanHuang): Do users also manually set exception attributes, and should we change FallbackError/InnerError to TypeError?
        if key == "__context__":
            if (
                isinstance(value, ConstantVariable)
                and value.get_py_value() is None
            ) or isinstance(
                value,
                ExceptionVariable,
            ):
                self.__context__ = value
            else:
                raise FallbackError(
                    f"`__context__` must be an ExceptionVariable, bug got {type(value)}:{value}"
                )
        elif key == "__cause__":
            if (
                isinstance(value, ConstantVariable)
                and value.get_py_value() is None
            ) or isinstance(
                value,
                ExceptionVariable,
            ):
                self.__cause__ = value
                self.__suppress_context__ = ConstantVariable.wrap_literal(
                    True, self.graph
                )
            else:
                raise FallbackError(
                    "exception cause must be None or derive from BaseException"
                )
        elif key == "__suppress_context__":
            if isinstance(value, ConstantVariable) and isinstance(
                value.get_py_value(), bool
            ):
                self.__suppress_context__ = value
            else:
                raise FallbackError("Type of __suppress_context__ must be bool")
        elif key == "__traceback__":
            if (
                isinstance(value, ConstantVariable)
                and value.get_py_value() is None
            ):
                self.__traceback__ = value
            else:
                raise FallbackError(
                    "Currently, SOT doesn't record information of __traceback__"
                )
        else:
            raise InnerError(f"ExceptionVariable don't need attribute {key}")

    def getattr(self, name: str, default=None) -> VariableBase:
        if name == "__traceback__":
            return ConstantVariable.wrap_literal(None, self.graph)

        if name == "args":
            return self.args_variables

        return super().getattr(name, default)

    @classmethod
    def check_if_exception_matches(
        cls,
        exc_instance: BuiltinVariable | ExceptionVariable,
        expected_exc_types: BuiltinVariable | TupleVariable,
    ):
        """
        try: exc_instance except: expected_exc_types
        """
        from .callable import BuiltinVariable
        from .container import TupleVariable

        if isinstance(expected_exc_types, TupleVariable):
            expected_types = expected_exc_types.get_wrapped_items()
        else:
            expected_types = [
                expected_exc_types,
            ]
        for expected_type in expected_types:
            if not isinstance(expected_type, BuiltinVariable):
                raise FallbackError(
                    f"`except ...` requires a BuiltinVariable as the exception type, but received: {expected_type}."
                )
            # Exception -> SotCapturedException
            expected_type_exception = SotCapturedExceptionFactory.get(
                expected_type.get_py_value()
            )
            if isinstance(exc_instance, ExceptionVariable) and issubclass(
                exc_instance.exc_type,
                expected_type_exception,
            ):
                return True
            elif isinstance(exc_instance, BuiltinVariable) and issubclass(
                exc_instance.get_py_value(), expected_type_exception
            ):
                return True
        return False

    @VariableFactory.register_from_value()
    def from_value(value: Exception, graph: FunctionGraph, tracker: Tracker):
        if isinstance(value, Exception):
            exception_var = ExceptionVariable(
                value, graph=graph, tracker=tracker
            )
            return exception_var
        return None


class EnumVariable(VariableBase):
    known_enum_classes = {}

    def __init__(
        self, value: Enum, graph: FunctionGraph = None, tracker: Tracker = None
    ) -> None:
        super().__init__(graph=graph, tracker=tracker)
        self.value = value

    def get_py_value(self, allow_tensor=False) -> Any:
        return self.value

    @VariableFactory.register_from_value()
    def from_value(value: Exception, graph: FunctionGraph, tracker: Tracker):
        if isinstance(value, Enum):
            var = EnumVariable(value, graph=graph, tracker=tracker)
            return var
        return None

    @check_faster_guard
    def make_faster_guard(self) -> list[paddle.framework.core.GuardNodeBase]:
        expr_node = self.tracker.guard_tree_expr_node()
        type_guard = paddle.framework.core.GuardNode(
            paddle.framework.core.TypeMatchGuard(self.get_py_type()),
            [expr_node],
        )
        value_guard = paddle.framework.core.GuardNode(
            paddle.framework.core.ValueMatchGuard(self.value),
            [expr_node],
        )
        return [type_guard, value_guard]

    @check_guard
    def make_stringified_guard(self) -> list[StringifiedExpression]:
        frame_value_tracer = self.tracker.trace_value_from_frame()
        enum_class = self.value.__class__
        class_name = enum_class.__name__
        enum_class_id = EnumVariable.get_enum_class_id(enum_class)
        extern_var_name = f"__{class_name}_{enum_class_id}"

        return [
            FasterStringifiedExpression(
                f"id(type({{}})) == {id(self.get_py_type())}",
                paddle.core.TypeMatchGuard(self.get_py_type()),
                [frame_value_tracer],
                union_free_vars(frame_value_tracer.free_vars),
            ),
            FasterStringifiedExpression(
                f"{{}} == {extern_var_name}.{self.value.name}",
                paddle.core.ValueMatchGuard(self.value),
                [frame_value_tracer],
                union_free_vars(
                    frame_value_tracer.free_vars,
                    {f"{extern_var_name}": self.get_py_value()},
                ),
            ),
        ]

    @classmethod
    def get_enum_class_id(cls, enum_class: type[Enum]):
        class_name = enum_class.__name__
        EnumVariable.known_enum_classes.setdefault(class_name, [])
        same_name_enums = EnumVariable.known_enum_classes[class_name]
        id = 0
        for i, cls in enumerate(same_name_enums):
            if enum_class == cls:
                id = i
                break
        else:
            id = len(same_name_enums)
            same_name_enums.append(enum_class)
        return id


class DataClassInstanceVariable(VariableBase):
    known_dataclasses = {}

    def __init__(
        self,
        data_dict: dict[str, Any],
        class_var: DataClassVariable,
        data_id: int | None,
        graph: FunctionGraph,
        tracker: Tracker,
    ):
        super().__init__(graph=graph, tracker=tracker)
        self.class_var = class_var
        self.proxy = self.graph.side_effects.get_proxy(
            MutableDictLikeData,
            data_dict,
            self.proxy_getter,
            id_getter=lambda _: data_id or id(data_dict),
        )

    def _reconstruct(self, codegen: PyCodeGen) -> None:
        codegen.gen_load_object(
            dataclass_from_dict,
            "___dataclass_from_dict",
        )
        self.getattr("__class__").reconstruct(codegen)
        data = self.proxy.get_all().keys()
        for key in data:
            if not isinstance(key, ConstTypes):
                raise InnerError(
                    f"[{self.__class__.__name__}] received {key} as key."
                )
            key_var = ConstantVariable.wrap_literal(key, self.graph)
            value_var = self.getattr(key)
            key_var.reconstruct(codegen)
            value_var.reconstruct(codegen)
        codegen.gen_build_map(len(data))
        codegen.gen_call_function(2)

    @cached_property
    def attr_proxy(self):
        return self.proxy

    def getattr(self, name: str, default=None):
        if name == "__class__":
            return self.class_var
        return self.proxy.get(name)

    def setattr(self, key: str, value):
        self.proxy.set(key, value)
        self.graph.side_effects.record_proxy_variable(self)
        return ConstantVariable.wrap_literal(None, self.graph)

    def proxy_getter(self, proxy: MutableDictLikeData, key: Any):
        if key not in proxy.original_data:
            return MutableDictLikeData.Empty()
        return VariableFactory.from_value(
            proxy.original_data[key],
            self.graph,
            tracker=GetAttrTracker(self, key, changed=proxy.has_changed),
        )

    def get_py_type(self):
        return self.class_var.get_py_value()

    def get_py_value(self, allow_tensor=False):
        return dataclass_from_dict(
            self.get_py_type(),
            {
                key: value.get_py_value(allow_tensor)
                for key, value in self.proxy.get_all().items()
            },
        )

    @check_faster_guard
    def make_faster_guard(self) -> list[paddle.framework.core.GuardNodeBase]:
        expr_node = self.tracker.guard_tree_expr_node()
        type_guard = paddle.framework.core.GuardNode(
            paddle.framework.core.TypeMatchGuard(self.get_py_type()),
            [expr_node],
        )
        guard_variables = filter(
            lambda var: not isinstance(var, MutableDictLikeData.Empty),
            self.proxy.reproduce(0).values(),
        )
        return reduce(
            operator.add,
            [[type_guard]]
            + [
                item.make_faster_guard()
                for item in guard_variables
                if item.tracker.need_guard()
            ],
        )

    @check_guard
    def make_stringified_guard(self) -> list[StringifiedExpression]:
        data_class = self.get_py_type()
        class_name = data_class.__name__
        data_class_id = DataClassInstanceVariable.get_class_id(data_class)
        extern_var_name = f"__{class_name}_{data_class_id}"
        frame_value_tracer = self.tracker.trace_value_from_frame()
        type_guard = FasterStringifiedExpression(
            f"isinstance({{}}, {extern_var_name})",
            paddle.framework.core.InstanceCheckGuard(self.get_py_type()),
            [frame_value_tracer],
            union_free_vars(
                frame_value_tracer.free_vars,
                {f"{extern_var_name}": self.get_py_type()},
            ),
        )
        guard_variables = filter(
            lambda var: not isinstance(var, MutableDictLikeData.Empty),
            self.proxy.reproduce(0).values(),
        )
        return reduce(
            operator.add,
            [[type_guard]]
            + [
                item.make_stringified_guard()
                for item in guard_variables
                if item.tracker.need_guard()
            ],
        )

    @classmethod
    def get_class_id(cls, data_class: type[Any]):
        class_name = data_class.__name__
        DataClassInstanceVariable.known_dataclasses.setdefault(class_name, [])
        same_name_dataclasses = DataClassInstanceVariable.known_dataclasses[
            class_name
        ]
        id = 0
        for i, cls in enumerate(same_name_dataclasses):
            if data_class == cls:
                id = i
                break
        else:
            id = len(same_name_dataclasses)
            same_name_dataclasses.append(data_class)
        return id

    def flatten_inner_vars(self) -> list[VariableBase]:
        return [
            self.getattr(fd.name)
            for fd in dataclasses.fields(self.get_py_type())
        ]

    @VariableFactory.register_from_value()
    def from_value(value: object, graph: FunctionGraph, tracker: Tracker):
        if is_dataclass_instance(value):
            class_var = VariableFactory.from_value(
                type(value), graph, DanglingTracker()
            )
            var = DataClassInstanceVariable(
                dataclass_as_dict(value),
                class_var,
                id(value),
                graph=graph,
                tracker=tracker,
            )
            class_var.tracker = GetAttrTracker(var, "__class__")
            return var
        return None
