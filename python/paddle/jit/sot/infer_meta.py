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

import copy
from functools import cached_property
from typing import TYPE_CHECKING, Any, TypeVar

import paddle
from paddle.amp.auto_cast import amp_state
from paddle.base.data_feeder import convert_dtype
from paddle.base.framework import convert_np_dtype_to_dtype_
from paddle.base.unique_name import (
    UniqueNameGenerator,
    guard as UniqueNameGuard,
)
from paddle.distributed.auto_parallel.placement_type import (
    get_shard_spec,
    to_placements,
)
from paddle.distributed.auto_parallel.static.dist_input_spec import (
    DistributedInputSpec,
)
from paddle.distributed.auto_parallel.static.utils import (
    convert_to_dims_mapping,
)
from paddle.framework import use_pir_api
from paddle.pir import is_fake_value
from paddle.static import InputSpec
from paddle.utils import flatten, is_sequence

from .symbolic_shape.symbolic_value import SymbolicInt
from .utils import (
    Cache,
    Singleton,
    map_if_extend,
    meta_str,
)
from .utils.exceptions import BreakGraphError, NullMetaBreak

if TYPE_CHECKING:
    import numpy.typing as npt

DynamicSymbolT = TypeVar("DynamicSymbolT")
SOT_INFER_META_INNER_VAR = "___SOT_INFER_META_INNER_VAR"


class DistInfo:
    def __init__(self, mesh=None, dims_mapping=None, local_shape=None):
        self.mesh = mesh
        self.dims_mapping = dims_mapping
        self.local_shape = local_shape

    @staticmethod
    def from_tensor(tensor: paddle.Tensor) -> DistInfo:
        assert (
            isinstance(tensor, paddle.Tensor) and tensor.is_dist()
        ), f"Expect a Tensor, but got a {type(tensor)}."

        mesh = tensor.process_mesh
        sharding_specs = get_shard_spec(
            mesh, tensor.placements, len(tensor.shape)
        )
        dims_mapping = convert_to_dims_mapping(sharding_specs, mesh)
        local_shape = tensor._local_value().shape
        return DistInfo(mesh, dims_mapping, local_shape)

    @staticmethod
    def from_value(value: paddle.pir.Value) -> DistInfo:
        assert (
            isinstance(value, paddle.pir.Value) and value.is_dist()
        ), f"Expect a Value, but got a {type(value)}."
        return DistInfo(
            value.dist_attr().process_mesh,
            value.dist_attr().dims_mapping,
            value._local_shape,
        )

    def __deepcopy__(self, memo):
        return DistInfo(
            mesh=copy.deepcopy(self.mesh),
            dims_mapping=copy.deepcopy(self.dims_mapping),
            local_shape=copy.deepcopy(self.local_shape),
        )

    def __repr__(self) -> str:
        return f"DistInfo(mesh={self.mesh}, dims_mapping={self.dims_mapping}, local_shape={self.local_shape})"


class MetaInfoOrNull:
    def __init__(self, meta: MetaInfo | None):
        self.meta = meta

    @staticmethod
    def null():
        return MetaInfoOrNull(None)

    def is_null(self):
        return self.meta is None

    def unwrap_or_breakgraph(self):
        if self.meta is None:
            raise BreakGraphError(NullMetaBreak())
        return self.meta

    def unwrap_unsafe(self):
        assert self.meta is not None, "MetaInfo is None"
        return self.meta

    def with_dynamic_axes(
        self, name: str, dynamic_axes: list[int]
    ) -> MetaInfoOrNull:
        if self.meta is None:
            return MetaInfoOrNull.null()
        return self.meta.with_dynamic_axes(name, dynamic_axes).wrap()

    def to_input_spec(self):
        if self.meta is None:
            return None
        return self.meta.to_input_spec()

    def guard_str(self):
        if self.meta is None:
            return "(Null)"
        return self.meta.guard_str()

    def __deepcopy__(self, memo):
        if self.meta is None:
            return MetaInfoOrNull(None)
        return MetaInfoOrNull(copy.deepcopy(self.meta))

    @staticmethod
    def _handle_legacy_ir_amp_dtype(dtype):
        # TODO(cleanup-legacy-ir) remove after pir become default state.
        # We always use float32 in simulation if AMP is enabled.
        if use_pir_api():
            return dtype
        assert isinstance(dtype, paddle.core.VarDesc.VarType)

        current_amp_state = amp_state()
        if (
            dtype == paddle.float16
            and current_amp_state is not None
            and current_amp_state["dtype"] == "float16"
        ):
            dtype = paddle.float32
        return dtype

    @staticmethod
    def from_tensor(
        tensor: paddle.Tensor, *, dynamic_axes: list[int] | None = None
    ) -> MetaInfoOrNull:
        if not tensor._is_dense_tensor_hold_allocation():
            return MetaInfoOrNull.null()
        assert isinstance(
            tensor, paddle.Tensor
        ), "Expect a Tensor, but got a Value."

        dtype = MetaInfoOrNull._handle_legacy_ir_amp_dtype(tensor.dtype)
        assert (
            -1 not in tensor.shape
        ), "Tensor shape should not contain -1, maybe you pass a Value to from_tensor"
        dynamic_axes = dynamic_axes or []
        shape = [
            SymbolicInt() if i in dynamic_axes else dim
            for i, dim in enumerate(tensor.shape)
        ]
        if tensor.is_dist():
            dist_info = DistInfo.from_tensor(tensor)
        else:
            dist_info = None
        return MetaInfo(
            shape,
            dtype,
            tensor.stop_gradient,
            tensor.name,
            tensor.persistable,
            tensor.type,
            tensor.place,
            None,
            dist_info=dist_info,
        ).wrap()

    @staticmethod
    def from_numpy(
        nparray: npt.NDArray[Any], *, dynamic_axes: list[int] | None = None
    ) -> MetaInfoOrNull:
        dtype = convert_np_dtype_to_dtype_(nparray.dtype)
        dynamic_axes = dynamic_axes or []
        shape = [
            SymbolicInt() if i in dynamic_axes else dim
            for i, dim in enumerate(nparray.shape)
        ]
        return MetaInfo(
            shape,
            dtype,
            True,  # stop_gradient
            None,
            None,  # persistable
            None,
            None,
            None,
            dist_info=None,
        ).wrap()

    @staticmethod
    def from_value(value) -> MetaInfoOrNull:
        if is_fake_value(value):
            return MetaInfoOrNull.null()
        name = SOT_INFER_META_INNER_VAR
        dtype = MetaInfoOrNull._handle_legacy_ir_amp_dtype(value.dtype)
        shape = [SymbolicInt() if dim == -1 else dim for dim in value.shape]
        for dim in shape:
            if isinstance(dim, int):
                assert dim >= 0, (
                    "Dimensions must be non-negative integers or SymbolicInt. "
                    f"Encountered value {dim} in shape {shape}."
                )

        if isinstance(value, paddle.pir.Value) and value.is_dist():
            dist_info = DistInfo.from_value(value)
        else:
            dist_info = None
        return MetaInfo(
            shape,
            dtype,
            value.stop_gradient,
            name,
            value.persistable,
            None,  # type is not a unified attribute in dygraph and static mode.
            None,  # We can't infer the right place in compile time.
            None,  # there's no spec_name specified when from_value.
            dist_info=dist_info,
        ).wrap()

    def __repr__(self):
        if self.meta is None:
            return "MetaInfoOrNull(None)"
        return f"MetaInfoOrNull({self.meta})"

    def __eq__(self, other):
        if self.meta is None:
            return other.meta is None
        if other.meta is None:
            return False
        return self.meta == other.meta

    def __hash__(self):
        if self.meta is None:
            return hash(None)
        return hash(self.meta)


class MetaInfo:
    shape: list[int | SymbolicInt]

    def __init__(
        self,
        shape,
        dtype,
        stop_gradient,
        name,
        persistable,
        type,
        place,
        spec_name=None,
        dist_info=None,
    ):
        assert (
            -1 not in shape
        ), "NOTE: Shape should not contain -1, consider convert it to SymbolicInt."
        self.name = name
        self.persistable = persistable
        self.type = type
        self.place = place
        self.shape = shape
        self.dtype = dtype
        self.stop_gradient = stop_gradient
        self.dist_info = dist_info
        self.spec_name = spec_name

    def wrap(self):
        return MetaInfoOrNull(self)

    def shape_with_special_symbol(
        self, dynamic_symbol: DynamicSymbolT = -1
    ) -> list[int | DynamicSymbolT]:
        return [
            dynamic_symbol if isinstance(dim, SymbolicInt) else dim
            for dim in self.shape
        ]

    def with_dynamic_axes(self, name: str, dynamic_axes: list[int]) -> MetaInfo:
        # NOTE(SigureMo): Make sure create a new shape list with dynamic axes.
        # We will create a new shape list variable lazily in the future.
        shape = [
            SymbolicInt(dim) if i in dynamic_axes else dim
            for i, dim in enumerate(self.shape)
        ]
        return MetaInfo(
            shape,
            self.dtype,
            self.stop_gradient,
            self.name,
            self.persistable,
            self.type,
            self.place,
            spec_name=name,
            dist_info=self.dist_info,
        )

    @property
    def dynamic_axes(self):
        return [
            i
            for i, dim in enumerate(self.shape)
            if isinstance(dim, SymbolicInt)
        ]

    def is_inner_var(self):
        return self.name == SOT_INFER_META_INNER_VAR

    def is_dynamic_shape(self):
        """
        if SymbolicInt in shape, return True
        else: return False
        """
        return len(self.dynamic_axes) > 0

    def to_input_spec(self) -> DistributedInputSpec | ConstrainedInputSpec:
        shape = self.shape_with_special_symbol(None)
        if self.dist_info is not None:
            placements = to_placements(
                self.dist_info.dims_mapping, self.dist_info.mesh
            )
            return DistributedInputSpec(
                shape,
                dtype=self.dtype,
                stop_gradient=self.stop_gradient,
                mesh=self.dist_info.mesh,
                placements=placements,
                local_shape=self.dist_info.local_shape,
            )
        else:
            return ConstrainedInputSpec(
                self.dynamic_axes,
                shape,
                dtype=self.dtype,
                name=self.spec_name,
                stop_gradient=self.stop_gradient,
            )

    def guard_str(self):
        shape = self.shape_with_special_symbol(SymbolicInt())
        return f"({shape}, {self.dtype}, {self.stop_gradient})"

    def __deepcopy__(self, memo):
        return MetaInfo(
            list(self.shape),
            self.dtype,
            self.stop_gradient,
            self.name,
            self.persistable,
            self.type,
            self.place,
            self.spec_name,
            dist_info=copy.deepcopy(self.dist_info),
        )

    def __repr__(self):
        return meta_str(self.shape, self.dtype, self.stop_gradient)

    def __eq__(self, meta):
        return (
            self.shape == meta.shape
            and self.dtype == meta.dtype
            and self.stop_gradient == meta.stop_gradient
        )

    def __hash__(self):
        return hash((tuple(self.shape), self.dtype, self.stop_gradient))


class VariableCreator(metaclass=Singleton):
    """
    We use the static graph Variable to infer the meta information of Tensor.
    This singleton class is used to create Variable for infer meta.
    """

    def __init__(self):
        # TODO(cleanup-legacy-ir): Remove the program and var_cache shims after PIR become default state.
        # self.var_cache = {}
        # self.main_program = paddle.static.Program()
        # self.startup_program = paddle.static.Program()
        self.var_name_generator = UniqueNameGenerator(SOT_INFER_META_INNER_VAR)

    def gen_name(self, meta_or_null: MetaInfoOrNull):
        if meta_or_null.is_null():
            return "null"
        meta = meta_or_null.unwrap_unsafe()
        name = f"{meta.dtype}_{meta.stop_gradient}_"
        name += "_".join(map(str, meta.shape))
        return name

    @property
    def var_cache(self):
        if paddle.framework.use_pir_api():
            return self.pir_var_cache
        else:
            return self.legacy_var_cache

    @cached_property
    def legacy_var_cache(self):
        return {}

    @cached_property
    def pir_var_cache(self):
        return {}

    @cached_property
    def legacy_programs(self):
        # Just for PIR and legacy IR compatibility.
        # This can be removed after PIR become default state.
        return (paddle.static.Program(), paddle.static.Program())

    @cached_property
    def pir_programs(self):
        return (paddle.static.Program(), paddle.static.Program())

    @property
    def main_program(self):
        if paddle.base.framework.use_pir_api():
            return self.pir_programs[0]
        else:
            return self.legacy_programs[0]

    @property
    def startup_program(self):
        if paddle.framework.use_pir_api():
            return self.pir_programs[1]
        else:
            return self.legacy_programs[1]

    def create_var(self, meta_or_null: MetaInfoOrNull):
        if meta_or_null.is_null():
            return None
        meta = meta_or_null.unwrap_unsafe()
        shape = meta.shape_with_special_symbol(-1)

        if paddle.framework.use_pir_api():
            with paddle.static.program_guard(
                self.main_program, self.startup_program
            ):
                var = paddle.static.input.data(
                    name=self.gen_name(meta.wrap()),
                    shape=shape,
                    dtype=convert_dtype(meta.dtype),
                )
                var.stop_gradient = meta.stop_gradient

                if meta.dist_info is not None:
                    mesh = meta.dist_info.mesh
                    placements = to_placements(
                        meta.dist_info.dims_mapping, mesh
                    )
                    var = paddle._pir_ops.shard_tensor(var, mesh, placements)
                    var.stop_gradient = meta.stop_gradient
        else:
            var = self.main_program.global_block().create_var(
                shape=shape,
                dtype=meta.dtype,
                stop_gradient=meta.stop_gradient,
            )
        assert not isinstance(
            var, paddle.Tensor
        ), "Expect a Variable, but got a Tensor."
        return var

    def get_variable(self, meta: MetaInfoOrNull, without_cache=False):
        var_feature_name = self.gen_name(meta)
        if without_cache:
            return self.create_var(meta)
        if var_feature_name not in self.var_cache:
            self.var_cache[var_feature_name] = self.create_var(meta)
        return self.var_cache[var_feature_name]

    def infer_meta(self, func, *args, **kwargs):
        with (
            paddle.base.framework._dygraph_guard(None),
            UniqueNameGuard(self.var_name_generator),
        ):
            if func is paddle.distributed.shard_tensor:
                args, kwargs = (
                    convert_meta_to_variable(args, without_cache=True),
                    convert_meta_to_variable(kwargs, without_cache=True),
                )
            else:
                args, kwargs = (
                    convert_meta_to_variable(args),
                    convert_meta_to_variable(kwargs),
                )

            with paddle.static.program_guard(
                self.main_program, self.startup_program
            ):
                if isinstance(func, str):
                    # TODO(Aurelius84): Is length of args always greater than 0?
                    # Do we need add condition check here?
                    func = getattr(args[0], func)
                    args = args[1:]
                out = func(*args, **kwargs)
        return convert_variable_to_meta_info(out)


def convert_meta_to_variable(args, without_cache=False):
    return map_if_extend(
        args,
        pred=lambda x: isinstance(x, MetaInfoOrNull),
        true_fn=lambda x: VariableCreator().get_variable(
            x, without_cache=without_cache
        ),
        false_fn=lambda x: x,
    )


def convert_meta_to_input_spec(args):
    return map_if_extend(
        args,
        pred=lambda x: isinstance(x, MetaInfoOrNull),
        true_fn=lambda x: x.to_input_spec(),
        # TODO(xiongkun): can x be tensor ?
        false_fn=lambda x: (
            paddle.static.InputSpec.from_tensor(x)
            if isinstance(x, paddle.Tensor)
            else x
        ),
    )


def convert_variable_to_meta_info(args):
    static_variable_type = (
        paddle.static.Variable
        if not paddle.base.framework.use_pir_api()
        else paddle.pir.Value
    )
    return map_if_extend(
        args,
        pred=lambda x: isinstance(x, static_variable_type),
        true_fn=lambda x: MetaInfoOrNull.from_value(x),
        false_fn=lambda x: x,
    )


def infer_meta(func, *args, **kwargs):
    fn = SpecialInferMeta().get_infermeta_fn(func)
    if fn:
        return fn(*args, **kwargs)
    return VariableCreator().infer_meta(func, *args, **kwargs)


def infer_meta_for_layer(layer, *args, **kwargs):
    assert isinstance(
        layer, paddle.nn.Layer
    ), f"Expect a Layer, but got {layer}."
    layer = paddle.jit.to_static(layer, full_graph=True)

    args_, kwargs_ = convert_meta_to_input_spec((args, kwargs))

    (
        concrete_program,
        partial_program_layer,
    ) = layer.forward.get_concrete_program(*args_, **kwargs_)

    if use_pir_api():
        output_values = partial_program_layer._outputs.var_list
    else:
        output_values = concrete_program.outputs

    out = partial_program_layer._restore_out(
        [
            x
            for x in paddle.utils.flatten(
                convert_variable_to_meta_info(output_values)
            )
            if isinstance(x, MetaInfoOrNull)
        ]
    )
    layer.forward.rollback()
    return out


def ast_infer_meta(static_function, *args, **kwargs):
    args_, kwargs_ = convert_meta_to_input_spec((args, kwargs))

    (
        concrete_program,
        partial_program_layer,
    ) = static_function.get_concrete_program(*args_, **kwargs_)

    out = partial_program_layer._restore_out(
        [
            x
            for x in paddle.utils.flatten(
                convert_variable_to_meta_info(concrete_program.outputs)
            )
            if isinstance(x, MetaInfoOrNull)
        ]
    )

    return out


class SpecialInferMeta(metaclass=Singleton):
    """
    There are some functions that cannot be inferred directly through static graph,
    and need to be implemented manually. This class is used to implement infer meta
    for these functions.
    """

    def __init__(self):
        pass

    def get_infermeta_fn(self, fn):
        try:
            funcname = fn.__name__
            return getattr(self, f"infermeta_{funcname}")
        except:
            pass
        return None

    def infermeta_grad(
        self,
        outputs,
        inputs,
        grad_outputs=None,
        retain_graph=None,
        create_graph=False,
        only_inputs=True,
        allow_unused=False,
        no_grad_vars=None,
    ):
        if not is_sequence(inputs):
            inputs = [inputs]
        return inputs


class InferMetaCache(Cache, metaclass=Singleton):
    def __init__(self):
        super().__init__(copy=True)

    def key_fn(
        self, func, *args, **kwargs
    ):  # args & kwargs have transformed to MetaInfo
        return (
            func,
            tuple(flatten(args)),
            tuple(kwargs.keys()),
            tuple(flatten(kwargs)),
        )

    def value_fn(self, func, *args, **kwargs):
        return infer_meta(func, *args, **kwargs)


class LayerInferMetaCache(Cache, metaclass=Singleton):
    def __init__(self):
        super().__init__(copy=True)

    def key_fn(self, layer, *args, **kwargs):
        params = [
            MetaInfoOrNull.from_tensor(x)
            for x in layer.parameters(include_sublayers=True)
        ]
        return (
            layer,
            tuple(params),
            tuple(flatten(args)),
            tuple(kwargs.keys()),
            tuple(flatten(kwargs)),
        )

    def value_fn(self, layer, *args, **kwargs):
        return infer_meta_for_layer(layer, *args, **kwargs)


class ConstrainedInputSpec(InputSpec):
    def __init__(self, dynamic_axes: list[int], *args, **kwargs):
        self.ranges: list[tuple[int, int | None, int | None]] = (
            []
        )  # (idx of dim, min, max)
        super().__init__(*args, **kwargs)
        for i in dynamic_axes:
            self.ranges.append((i, 1, None))
