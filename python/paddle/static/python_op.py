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

import inspect
import sys
import types
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property, partial, wraps
from typing import Any, Callable, Generic, TypeVar, overload

from typing_extensions import ParamSpec

import paddle
from paddle import _C_ops

HAS_VAR_ARGS_OR_KWARGS: int = inspect.CO_VARARGS | inspect.CO_VARKEYWORDS


P1 = ParamSpec("P1")
R1 = TypeVar("R1")


class MissingArgument:
    def __init__(self, fn: Callable[P1, R1], name: str):
        self.fn = fn
        self.name = name

    def __repr__(self):
        return f"<Required parameter '{self.name}' for function {self.fn.__name__}>"


def extract_default(fn: Callable[P1, R1], parameter: inspect.Parameter):
    if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
        return ()
    elif parameter.kind is inspect.Parameter.VAR_KEYWORD:
        return {}
    elif parameter.default is inspect.Parameter.empty:
        return MissingArgument(fn, parameter.name)
    return parameter.default


def get_fn_defaults_params(fn: Callable[P1, R1]) -> tuple:
    fn_defaults_params = [
        extract_default(fn, param)
        for param in inspect.signature(fn).parameters.values()
    ]
    for i, default in enumerate(fn_defaults_params):
        if not isinstance(default, MissingArgument):
            fn_defaults_params = fn_defaults_params[i:]
            break
    return tuple(fn_defaults_params)


def eliminate_positional_or_keyword_only(
    fn: Callable[P1, R1],
) -> Callable[P1, R1]:
    assert isinstance(fn, types.FunctionType), "Only support regular function"
    code = fn.__code__
    co_flags: int = code.co_flags & ~HAS_VAR_ARGS_OR_KWARGS

    argcount = (
        code.co_argcount
        + code.co_kwonlyargcount
        + bool(code.co_flags & inspect.CO_VARARGS)
        + bool(code.co_flags & inspect.CO_VARKEYWORDS)
    )
    if sys.version_info >= (3, 11):
        new_code = types.CodeType(
            argcount,  # co_argcount
            0,  # posonlyargcount, eliminated
            0,  # kwonlyargcount, eliminated
            code.co_nlocals,
            code.co_stacksize,
            co_flags,
            code.co_code,
            code.co_consts,
            code.co_names,
            code.co_varnames,
            code.co_filename,
            code.co_name,
            code.co_qualname,
            code.co_firstlineno,
            code.co_linetable,
            code.co_exceptiontable,
            code.co_freevars,
            code.co_cellvars,
        )
    else:
        new_code = types.CodeType(
            argcount,  # co_argcount
            0,  # posonlyargcount, eliminated
            0,  # kwonlyargcount, eliminated
            code.co_nlocals,
            code.co_stacksize,
            co_flags,
            code.co_code,
            code.co_consts,
            code.co_names,
            code.co_varnames,
            code.co_filename,
            code.co_name,
            code.co_firstlineno,
            code.co_linetable
            if sys.version_info >= (3, 10)
            else code.co_lnotab,
            code.co_freevars,
            code.co_cellvars,
        )

    fn_defaults_params = get_fn_defaults_params(fn)
    new_fn = types.FunctionType(
        new_code,
        fn.__globals__,
        fn.__name__,
        fn_defaults_params,
        fn.__closure__,
    )
    new_fn.__name__ = fn.__name__
    new_fn.__doc__ = fn.__doc__
    new_fn.__annotations__ = fn.__annotations__
    new_fn.__kwdefaults__ = None  # already merged into defaults
    return new_fn


@dataclass
class FunctionPack(Generic[P1, R1]):
    fn: Callable[P1, R1]
    infer_meta: Callable[..., Any]

    def id(self) -> int:
        return id(self.fn)


class ConstantParams:
    def __init__(self, params: dict[str, Any]):
        self.params = params

    def __hash__(self):
        return custom_hash(self.params)

    def __eq__(self, other):
        if not isinstance(other, ConstantParams):
            return False
        return self.params == other.params


@dataclass
class OriginalFunctionPack(FunctionPack[P1, R1]):
    def __post_init__(self):
        self._specialized_fns: dict[ConstantParams, FunctionPack[P1, R1]] = {}

    @cached_property
    def fn_eliminated(self) -> Callable[P1, R1]:
        return eliminate_positional_or_keyword_only(self.fn)

    @cached_property
    def infer_meta_eliminated(self) -> Callable[..., Any]:
        return eliminate_positional_or_keyword_only(self.infer_meta)

    def get_bound_args(self, /, *args: P1.args, **kwargs: P1.kwargs):
        sig = inspect.signature(self.fn)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        return bound_args.arguments

    def separate_mutable_and_const_params(
        self, /, *args: P1.args, **kwargs: P1.kwargs
    ) -> tuple[dict[str, paddle.pir.Value], dict[str, Any]]:
        params = self.get_bound_args(*args, **kwargs)

        mutable_params = {}
        const_params = {}

        # TODO: Support container types like list, dict, tuple
        for k, v in params.items():
            if isinstance(v, paddle.pir.Value):
                mutable_params[k] = v
            else:
                const_params[k] = v

        return mutable_params, const_params

    def specialize(self, const_params: dict[str, Any]) -> FunctionPack[P1, R1]:
        const_params_wrapper = ConstantParams(const_params)
        if const_params_wrapper in self._specialized_fns:
            return self._specialized_fns[const_params_wrapper]

        specialized_fn = partial(self.fn_eliminated, **const_params)
        specialized_infer_meta = partial(
            self.infer_meta_eliminated, **const_params
        )
        specialized_fn_pack = FunctionPack(
            specialized_fn, specialized_infer_meta
        )
        self._specialized_fns[const_params_wrapper] = specialized_fn_pack
        return specialized_fn_pack


class FunctionRegistry:
    def __init__(self):
        self._registry: dict[str, OriginalFunctionPack[Any, Any]] = {}

    def register(
        self,
        name: str,
        fn: Callable[P1, R1],
        infer_meta: Callable[..., Any],
    ):
        if name not in self._registry:
            self._registry[name] = OriginalFunctionPack(fn, infer_meta)
            return self._registry[name]
        fn_pack = self._registry[name]
        if fn is not fn_pack.fn or infer_meta is not fn_pack.infer_meta:
            raise ValueError(
                f"Function '{name}' is already registered with a different implementation."
            )
        return fn_pack

    def get(self, name: str) -> OriginalFunctionPack[Any, Any]:
        if name not in self._registry:
            raise KeyError(f"Function '{name}' is not registered.")
        return self._registry[name]


FUNCTION_REGISTRY = FunctionRegistry()


def bind_constants(fn, infer_meta, *args, **kwargs):
    sig = inspect.signature(fn)
    bound_args = sig.bind(*args, **kwargs)
    bound_args.apply_defaults()
    params = bound_args.arguments

    mutable_params = {}
    const_params = {}

    for k, v in params.items():
        if isinstance(v, paddle.pir.Value):
            mutable_params[k] = v
        else:
            const_params[k] = v

    mutable_arg_names = list(mutable_params.keys())
    fn = eliminate_positional_or_keyword_only(fn)
    infer_meta = eliminate_positional_or_keyword_only(infer_meta)
    return (
        mutable_arg_names,
        partial(fn, **const_params),
        partial(infer_meta, **const_params),
        list(mutable_params.values()),
        const_params,
    )


def run_in_dynamic_mode(fn):
    def dynamic_mode_fn(*args, **kwargs):
        with paddle.base.dygraph.base.guard():
            return fn(*args, **kwargs)

    return dynamic_mode_fn


def custom_hash(obj):
    # Compute a hash for various types of objects, including unhashable ones.
    # This may not be collision-free. For example, hash(-1) is same as hash(-2).
    # We use dict to resolve collisions in ConstantParams.

    # Handle basic types
    if isinstance(obj, (int, float, str, bool, bytes)):
        return hash(obj)

    # Handle sequences (like list, tuple, set, frozenset)
    if isinstance(obj, (Sequence, frozenset, set)):
        type_id_map = {list: 1, tuple: 2, frozenset: 3, set: 4}
        type_id = type_id_map.get(type(obj), 0)
        return hash((type_id, *tuple(custom_hash(item) for item in obj)))

    # Handle mappings (like dict)
    if isinstance(obj, Mapping):
        type_id = 5
        items_hashed = tuple(
            sorted((custom_hash(k), custom_hash(v)) for k, v in obj.items())
        )
        return hash((type_id, *items_hashed))

    # Fallback: try to use the built-in hash, or use id() if unhashable
    try:
        return hash(obj)
    except TypeError:
        return id(obj)


@overload
def register_op(
    fn: Callable[P1, R1],
    /,
    *,
    name: str | None = None,
    infer_meta: Callable[..., Any] | None = None,
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
    inplace_map: dict[str, str] | None = None,
) -> Callable[P1, R1]: ...


@overload
def register_op(
    fn: None = None,
    /,
    *,
    name: str | None = None,
    infer_meta: Callable[..., Any] | None = None,
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
    inplace_map: dict[str, str] | None = None,
) -> Callable[[Callable[P1, R1]], Callable[P1, R1]]: ...


def register_op(
    fn: Callable[P1, R1] | None = None,
    /,
    *,
    name: str | None = None,
    infer_meta: Callable[..., Any] | None = None,
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
    inplace_map: dict[str, str] | None = None,
):
    if input_names is None:
        raise ValueError("Currently, input_names must be provided.")
    if output_names is None:
        raise ValueError("Currently, output_names must be provided.")
    if infer_meta is None:
        raise ValueError("Currently, infer_meta must be provided.")

    def _register_op(
        real_fn: Callable[P1, R1],
    ) -> Callable[P1, R1]:
        op_name = name or real_fn.__name__

        @paddle.jit.marker.unified
        @wraps(real_fn)
        def wrapped_fn(*args: P1.args, **kwargs: P1.kwargs) -> R1:
            if paddle.in_dynamic_mode():
                return real_fn(*args, **kwargs)

            fn_pack = FUNCTION_REGISTRY.register(op_name, real_fn, infer_meta)
            mutable_params, const_params = (
                fn_pack.separate_mutable_and_const_params(*args, **kwargs)
            )
            specialized_fn_pack = fn_pack.specialize(const_params)

            assert len(mutable_params) == len(input_names), (
                f"Number of mutable arguments ({len(mutable_params)}) does not match "
                f"the number of input names ({len(input_names)})."
            )

            out = _C_ops._run_python_op(
                *mutable_params.values(),
                name=f"{op_name}_{specialized_fn_pack.id()}",
                input_names=input_names,
                output_names=output_names,
                attrs={
                    "infer_meta_fn_ptr": specialized_fn_pack.infer_meta,
                    "fn_ptr": run_in_dynamic_mode(specialized_fn_pack.fn),
                },
                inplace_map=inplace_map or {},
            )

            return out[0] if len(output_names) == 1 else out

        return wrapped_fn

    # Handle @register_op(...)
    if fn is None:
        return _register_op
    # Handle @register_op
    return _register_op(fn)
