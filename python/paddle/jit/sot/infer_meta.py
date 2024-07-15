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

from functools import cached_property
from typing import TypeVar

from typing_extensions import Self

import paddle
from paddle.amp.auto_cast import amp_state
from paddle.base.data_feeder import convert_dtype
from paddle.base.unique_name import (
    UniqueNameGenerator,
    guard as UniqueNameGuard,
)
from paddle.framework import use_pir_api
from paddle.utils import flatten, is_sequence

from .utils import Cache, Singleton, map_if_extend, meta_str

DynamicSymbolT = TypeVar("DynamicSymbolT")


class SymbolicInt(metaclass=Singleton):
    def __repr__(self) -> str:
        return "SymbolicInt()"

    def __str__(self) -> str:
        return "SymbolicInt()"


class MetaInfo:
    def __init__(
        self,
        shape,
        dtype,
        stop_gradient,
        name,
        persistable,
        type,
        place,
    ):
        assert (
            -1 not in shape
        ), "NOTE: Shape should not contain -1, consider convert it to SymbolicInt."
        self.name = name
        self.persistable = persistable
        self.type = type
        self.place = place
        self.shape: list[int | SymbolicInt] = shape
        self.dtype = dtype
        self.stop_gradient = stop_gradient

    def shape_with_special_symbol(
        self, dynamic_symbol: DynamicSymbolT = -1
    ) -> list[int | DynamicSymbolT]:
        return [
            dim if not isinstance(dim, SymbolicInt) else dynamic_symbol
            for dim in self.shape
        ]

    def with_dynamic_axes(self, dynamic_axes: list[int]) -> Self:
        shape = [
            SymbolicInt() if i in dynamic_axes else dim
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
        )

    @property
    def dynamic_axes(self):
        return [
            i
            for i, dim in enumerate(self.shape)
            if isinstance(dim, SymbolicInt)
        ]

    @staticmethod
    def _handle_legacy_ir_amp_dtype(dtype):
        expected_dtype_class = (
            paddle.core.DataType
            if paddle.framework.use_pir_api()
            else paddle.core.VarDesc.VarType
        )
        assert isinstance(dtype, expected_dtype_class)

        # TODO(@xiongkun) remove after pir become default state.
        # We always use float32 in simulation if AMP is enabled.
        current_amp_state = amp_state()
        if (
            not use_pir_api()
            and dtype == paddle.float16
            and current_amp_state is not None
            and current_amp_state["dtype"] == "float16"
        ):
            dtype = paddle.float32
        return dtype

    @staticmethod
    def from_tensor(
        tensor: paddle.Tensor, *, dynamic_axes: list[int] | None = None
    ) -> MetaInfo:
        assert isinstance(
            tensor, paddle.Tensor
        ), "Expect a Tensor, but got a Value."

        dtype = MetaInfo._handle_legacy_ir_amp_dtype(tensor.dtype)
        assert (
            -1 not in tensor.shape
        ), "Tensor shape should not contain -1, maybe you pass a Value to from_tensor"
        dynamic_axes = dynamic_axes or []
        shape = [
            SymbolicInt() if i in dynamic_axes else dim
            for i, dim in enumerate(tensor.shape)
        ]
        return MetaInfo(
            shape,
            dtype,
            tensor.stop_gradient,
            tensor.name,
            tensor.persistable,
            tensor.type,
            tensor.place,
        )

    @staticmethod
    def from_value(value) -> MetaInfo:
        if isinstance(value, paddle.pir.Value):
            name = "Value@NoName"
        else:
            name = value.name
        dtype = MetaInfo._handle_legacy_ir_amp_dtype(value.dtype)
        shape = [SymbolicInt() if dim == -1 else dim for dim in value.shape]
        return MetaInfo(
            shape,
            dtype,
            value.stop_gradient,
            name,
            value.persistable,
            value.type,
            value.place,
        )

    def is_dynamic_shape(self):
        """
        if SymbolicInt in shape, return True
        else: return False
        """
        return len(self.dynamic_axes) > 0

    def to_input_spec(self):
        shape = self.shape_with_special_symbol(None)
        return paddle.static.InputSpec(
            shape, dtype=self.dtype, stop_gradient=self.stop_gradient
        )

    def guard_str(self):
        shape = self.shape_with_special_symbol(SymbolicInt())
        return f"({shape}, {self.dtype}, {self.stop_gradient})"

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
        self.var_name_generator = UniqueNameGenerator("infer_meta_variable_")

    def gen_name(self, meta):
        name = f"{meta.dtype}_{meta.stop_gradient}"
        for l in meta.shape:
            name += f"_{l}"
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

    def create_var(self, meta: MetaInfo):
        shape = meta.shape_with_special_symbol(-1)

        if paddle.framework.use_pir_api():
            with paddle.static.program_guard(
                self.main_program, self.startup_program
            ):
                var = paddle.static.input.data(
                    name=self.gen_name(meta),
                    shape=shape,
                    dtype=convert_dtype(meta.dtype),
                )
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

    def get_variable(self, meta):
        var_feature_name = self.gen_name(meta)
        if var_feature_name not in self.var_cache:
            self.var_cache[var_feature_name] = self.create_var(meta)
        return self.var_cache[var_feature_name]

    def infer_meta(self, func, *args, **kwargs):
        with paddle.base.framework._dygraph_guard(None), UniqueNameGuard(
            self.var_name_generator
        ):
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
                    out = getattr(args[0], func)(*args[1:], **kwargs)
                else:
                    out = func(*args, **kwargs)

        return convert_variable_to_meta_info(out)


def convert_meta_to_variable(args):
    return map_if_extend(
        args,
        pred=lambda x: isinstance(x, MetaInfo),
        true_fn=lambda x: VariableCreator().get_variable(x),
        false_fn=lambda x: x,
    )


def convert_meta_to_input_spec(args):
    return map_if_extend(
        args,
        pred=lambda x: isinstance(x, MetaInfo),
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
        true_fn=lambda x: MetaInfo.from_value(x),
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
            if isinstance(x, MetaInfo)
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
            if isinstance(x, MetaInfo)
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
    def key_fn(
        self, func, *args, **kwargs
    ):  # args & kwargs have transformed to MetaInfo
        try:
            retval = hash(
                (
                    func,
                    tuple(flatten(args)),
                    tuple(kwargs.keys()),
                    tuple(flatten(kwargs)),
                )
            )
        except Exception as e:
            return None
        return retval

    def value_fn(self, func, *args, **kwargs):
        return infer_meta(func, *args, **kwargs)


class LayerInferMetaCache(Cache, metaclass=Singleton):
    def key_fn(self, layer, *args, **kwargs):
        params = [
            MetaInfo.from_value(x)
            for x in layer.parameters(include_sublayers=True)
        ]
        try:
            retval = hash(
                (
                    layer,
                    tuple(params),
                    tuple(flatten(args)),
                    tuple(kwargs.keys()),
                    tuple(flatten(kwargs)),
                )
            )
        except Exception as e:
            return None
        return retval

    def value_fn(self, layer, *args, **kwargs):
        return infer_meta_for_layer(layer, *args, **kwargs)
