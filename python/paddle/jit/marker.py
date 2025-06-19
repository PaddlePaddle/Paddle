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
from typing import (
    Callable,
    Protocol,
    TypeVar,
    overload,
)

from typing_extensions import (
    ParamSpec,
)

import paddle

from .dy2static.utils import (
    TransformOptions,
)

_RetT = TypeVar("_RetT")
_InputT = ParamSpec("_InputT")


class _NotToStaticDecorator(Protocol):
    @overload
    def __call__(
        self, func: Callable[_InputT, _RetT]
    ) -> Callable[_InputT, _RetT]: ...

    @overload
    def __call__(self, func: None = ...) -> _NotToStaticDecorator: ...


@overload
def not_to_static(
    func: Callable[_InputT, _RetT],
) -> Callable[_InputT, _RetT]: ...


@overload
def not_to_static(func: None = ...) -> _NotToStaticDecorator: ...


# Legacy decorator only for AST
def not_to_static(func=None):
    """
    A Decorator to suppresses the convention of a function.

    Args:
        func(callable): The function to decorate.

    Returns:
        callable: A function which won't be converted in Dynamic-to-Static.

    Examples:
        .. code-block:: python

            >>> # doctest: +SKIP('`paddle.jit.to_static` can not run in xdoctest')
            >>> import paddle

            >>> @paddle.jit.not_to_static
            ... def func_not_to_static(x):
            ...     res = x - 1
            ...     return res

            >>> @paddle.jit.to_static
            ... def func(x):
            ...     if paddle.mean(x) < 0:
            ...         out = func_not_to_static(x)
            ...     else:
            ...         out = x + 1
            ...     return out
            ...
            >>> x = paddle.ones([1, 2], dtype='float32')
            >>> out = func(x)
            >>> print(out)
            Tensor(shape=[1, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[2., 2.]])
    """
    return unified(func, for_sot=False, for_ast=True)


def unified(
    fn: Callable[_InputT, _RetT] | type[paddle.nn.Layer] | None = None,
    *,
    for_sot: bool = True,
    for_ast: bool = True,
) -> Callable[_InputT, _RetT]:
    """
    Mark a function already unified in dygraph and static mode. So
    that it won't be transformed again in SOT or AST mode.

    Args:
        fn(callable): The function to decorate.
        for_sot(bool): Whether to mark the function as unified in SOT mode.
        for_ast(bool): Whether to mark the function as unified in AST mode.
    """

    def _mark_as_unified(fn, *, for_sot: bool, for_ast: bool):
        mode = TransformOptions.ToStaticMode.Nil()
        if for_sot:
            mode |= TransformOptions.ToStaticMode.SOT
        if for_ast:
            mode |= TransformOptions.ToStaticMode.AST
        options = TransformOptions(
            skip_transform_mode=mode,
        )
        options.attach(fn)
        return fn

    if fn is None:
        return lambda fn: _mark_as_unified(fn, for_sot=for_sot, for_ast=for_ast)
    return _mark_as_unified(fn, for_sot=for_sot, for_ast=for_ast)


def force_dynamic(
    fn: Callable[_InputT, _RetT] | type[paddle.nn.Layer] | None = None,
) -> Callable[_InputT, _RetT]:
    """
    Mark a function or paddle.nn.Layer to be executed in dynamic mode, it will
    break the graph and prevent it from being converted to static mode.
    """
    from paddle.jit import sot

    if inspect.isclass(fn) and issubclass(fn, paddle.nn.Layer):
        sot.utils.paddle_api_config.add_break_graph_layer_class(fn)
        return fn
    if inspect.isfunction(fn):
        sot.utils.paddle_api_config.add_break_graph_function(fn)
        return fn

    raise TypeError(
        f"Expected a callable or paddle.nn.Layer, but got {type(fn).__name__}."
    )
