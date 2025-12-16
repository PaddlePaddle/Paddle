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
import types
from collections.abc import Sequence
from functools import partial, wraps
from typing import Any, Callable, ParamSpec, TypeVar, overload

import paddle
from paddle import _C_ops

# from paddle.static.meta_tensor import MetaTensorWrapper

HAS_ARGS_OR_KWARGS: int = inspect.CO_VARARGS | inspect.CO_VARKEYWORDS


P1 = ParamSpec("P1")
R1 = TypeVar("R1")
P2 = ParamSpec("P2")
R2 = TypeVar("R2")


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


def eliminate_positional_only(fn: Callable[P1, R1]) -> Callable[P1, R1]:
    code = fn.__code__
    co_flags: int = code.co_flags & ~HAS_ARGS_OR_KWARGS
    co_flags = code.co_flags

    # TODO: currently, only support Python3.10
    if hasattr(code, "co_posonlyargcount"):
        argcount = (
            code.co_argcount
            + code.co_kwonlyargcount
            + bool(code.co_flags & inspect.CO_VARARGS)
            + bool(code.co_flags & inspect.CO_VARKEYWORDS)
        )

        new_code = types.CodeType(
            argcount,  # co_argcount
            0,  # posonlyargcount
            0,  # kwonlyargcount
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
            code.co_lnotab,
            code.co_freevars,
            code.co_cellvars,
        )
    else:
        raise ValueError

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
    new_fn.__kwdefaults__ = None
    return new_fn


def bind_constants(fn, infer_meta, input_names, *args, **kwargs):
    sig = inspect.signature(fn)
    bound_args = sig.bind(*args, **kwargs)
    bound_args.apply_defaults()
    params = bound_args.arguments

    # NOTE: dict 可以保留顺序
    mutable_params = {}
    const_params = {}

    for k, v in params.items():
        if isinstance(v, paddle.pir.Value):
            mutable_params[k] = v
        else:
            const_params[k] = v

    mutable_arg_names = list(mutable_params.keys())
    fn = eliminate_positional_only(fn)
    infer_meta = eliminate_positional_only(infer_meta)
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
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                print(e)
                raise

    return dynamic_mode_fn


from collections.abc import Mapping


def custom_hash(obj):
    # TODO: Check a case
    # hash(-1) == hash(-2)
    if isinstance(obj, (int, float, str, bool, bytes)):
        return hash(obj)

    # 2. 已知可哈希容器 (如 tuple, frozenset)
    if isinstance(obj, (tuple, frozenset)):
        try:
            return hash(obj)
        except TypeError:
            pass

    # 3. 不可哈希的 Sequence (如 list)
    if isinstance(obj, (Sequence, set)):
        # 使用 (0, ...) 前缀是为了区分列表和元组的哈希值
        try:
            return hash((0, *tuple(custom_hash(item) for item in obj)))
        except TypeError:
            pass

    # 4. 不可哈希的 Mapping (如 dict)
    if isinstance(obj, Mapping):
        # 使用 (1, ...) 前缀是为了区分字典和其他容器的哈希值
        try:
            items_hashed = tuple(
                sorted((custom_hash(k), custom_hash(v)) for k, v in obj.items())
            )
            return hash((1, *items_hashed))
        except TypeError:
            pass

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
    inplace_map: list[str, str] | None = None,
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
    inplace_map: list[str, str] | None = None,
) -> Callable[[Callable[P1, R1]], Callable[P1, R1]]: ...


def register_op(
    fn: Callable[P1, R1] | None = None,
    /,
    *,
    name: str | None = None,
    infer_meta: Callable[..., Any] | None = None,
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
    inplace_map: list[str, str] | None = None,
):
    """
    注册算子的装饰器，支持传入元数据推导函数和输入输出配置。
    """

    # 内部装饰器逻辑
    def _register_op(
        real_fn: Callable[P1, R1],
    ) -> Callable[P1, R1]:
        op_name = name or real_fn.__name__

        @paddle.jit.marker.unified
        @wraps(real_fn)
        def wrapped_fn(*args: P1.args, **kwargs: P1.kwargs) -> R1:
            if paddle.in_dynamic_mode():
                return real_fn(*args, **kwargs)

            (
                mutable_arg_names,
                bound_constants_fn,
                bound_constants_infer_meta,
                args,
                const_params,
            ) = bind_constants(
                real_fn, infer_meta, input_names, *args, **kwargs
            )
            assert len(mutable_arg_names) == len(input_names), (
                f"{mutable_arg_names=} != {input_names=}"
            )

            const_params_hash = custom_hash(const_params)

            # 调用底层算子运行逻辑
            out = _C_ops._run_custom_pyop(
                *args,  # kwargs 还需要吗？
                name=f"{op_name}_{const_params_hash}",  # 每次绑定一次，说明绑定的
                # inputs=inputs,
                input_names=input_names,
                output_names=output_names,
                attrs={
                    "infer_meta_fn_ptr": bound_constants_infer_meta,
                    "fn_ptr": run_in_dynamic_mode(bound_constants_fn),
                },
                inplace_map=inplace_map or {},
            )

            return out[0] if len(output_names) == 1 else out

        return wrapped_fn

    # 处理装饰器调用的两种方式：
    # 1. @register_op(...) -> fn is None
    # 2. @register_op -> fn is not None (不带括号，但在本例中不适用，因为必须传参)
    if fn is None:
        return _register_op
    return _register_op(fn)
