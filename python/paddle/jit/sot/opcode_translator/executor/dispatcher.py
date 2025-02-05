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
import inspect
import operator
from functools import cached_property, reduce
from typing import TYPE_CHECKING, Any, Callable, Dict, Tuple, TypeVar

from ...utils import InnerError, NameGenerator, hashable

if TYPE_CHECKING:
    T = TypeVar("T")
    Args = Tuple[T, ...]
    Kwargs = Dict[str, T]


def format_type(type_: type[Any] | tuple[type[Any], ...]) -> str:
    if not isinstance(type_, tuple):
        type_ = (type_,)
    return " | ".join([t.__name__ for t in type_])


def format_param(param: Parameter) -> str:
    kind = param.kind
    if kind == inspect.Parameter.VAR_POSITIONAL:
        return f"*{format_type(param.type)}"
    elif kind == inspect.Parameter.VAR_KEYWORD:
        return f"**{format_type(param.type)}"
    else:
        return format_type(param.type)


def convert_annotation_to_type(type_str: str) -> tuple[type[Any], ...]:
    """
    Convert type annotation to runtime value. Because we are using :pep:`563`
    to use the future annotation syntax, we cannot use `get_type_hints <https://docs.python.org/3.8/library/typing.html#typing.get_type_hints>`_
    directly. Currently, only the builtins and variables namespaces are supported.

    Returns:
        tuple: The converted type.
    """

    import builtins

    from . import variables

    type_str = type_str.strip()
    if type_str == "Any":
        type_str = "object"

    if "|" in type_str:
        return reduce(
            operator.add, map(convert_annotation_to_type, type_str.split("|"))
        )

    search_namespaces = [variables, builtins]
    for namespace in search_namespaces:
        if hasattr(namespace, type_str):
            return (getattr(namespace, type_str),)
    raise InnerError(f"Cannot find type {type_str} in {search_namespaces}")


class Parameter:
    name_gen = NameGenerator("param_")
    annotation: str
    name: str

    def __init__(
        self,
        annotation: str,
        *,
        kind: inspect._ParameterKind = inspect.Parameter.POSITIONAL_OR_KEYWORD,
        name: str | None = None,
        default: Any = inspect._empty,
    ):
        self.name = name if name is not None else Parameter.name_gen.next()
        self.annotation = annotation
        self.kind = kind
        self.default = default

    def to_parameter(self) -> inspect.Parameter:
        return inspect.Parameter(
            self.name,
            kind=self.kind,
            annotation=self.annotation,
            default=copy.copy(self.default),
        )

    @cached_property
    def type(self) -> tuple[type[Any], ...]:
        return convert_annotation_to_type(self.annotation)

    def match_arg(self, arg: Any) -> bool:
        # TODO: support VAR_KEYWORD
        if self.kind == inspect.Parameter.VAR_POSITIONAL:
            is_tuple = isinstance(arg, tuple)
            return is_tuple and all(isinstance(a, self.type) for a in arg)
        elif self.kind == inspect.Parameter.VAR_KEYWORD:
            is_dict = isinstance(arg, dict)
            return is_dict and all(
                isinstance(a, self.type) for a in arg.values()
            )
        else:
            return isinstance(arg, self.type)

    @staticmethod
    def from_str(annotation: str) -> Parameter:
        return Parameter(annotation)

    @staticmethod
    def from_parameter(parameter: inspect.Parameter) -> Parameter:
        if parameter.annotation != parameter.empty and not isinstance(
            parameter.annotation, str
        ):
            raise InnerError(
                f"Parameter {parameter} has annotation {parameter.annotation} "
                "which is not a string. Please add `from __future__ import annotations` "
                "to the top of your file."
            )
        annotation = (
            parameter.annotation
            if parameter.annotation != parameter.empty
            else "Any"
        )

        return Parameter(
            annotation,
            kind=parameter.kind,
            name=parameter.name,
            default=parameter.default,
        )

    def __repr__(self) -> str:
        default_repr = f"= {self.default!r}"
        return f"Parameter({', '.join([self.annotation, default_repr])})"


def optional(annotation: str, default: Any = None) -> Parameter:
    return Parameter(annotation, default=default)


class Pattern:
    parameters: dict[str, Parameter]
    signature: inspect.Signature

    def __init__(
        self,
        *parameters: Parameter,
    ):
        self.parameters = {
            parameter.name: parameter for parameter in parameters
        }
        self.signature = inspect.Signature(
            [parameter.to_parameter() for parameter in self.parameters.values()]
        )

    def match_inputs(self, /, *args: Any, **kwargs: Any) -> bool:
        """
        Match the input parameters of the function.

        Returns:
            bool: Whether the input parameters match the pattern.
        """
        try:
            bound_args = self.signature.bind(*args, **kwargs)
        except TypeError:
            return False
        for arg_name, arg_value in bound_args.arguments.items():
            if arg_name not in self.parameters:
                continue
            if not self.parameters[arg_name].match_arg(arg_value):
                return False
        return True

    def __repr__(self) -> str:
        types_repr = ", ".join(
            [format_param(param) for param in self.parameters.values()]
        )
        return f"Pattern({types_repr})"


class Dispatcher:
    """
    Used for pattern registration and distribution.

    For more design ideas, refer to the `Builtin dispatcher <https://github.com/PaddlePaddle/PaddleSOT/blob/develop/docs/design/builtin-dispatcher.md>`_ for details.

    Examples:

        >>> def builtin_add(a: int, b: int) -> int:
        ...     ...
        ...
        >>> Dispatcher.register(builtin_add, ("int", "int"), lambda a, b: a + b)
        >>> handler = Dispatcher.dispatch(builtin_add, 1, 2)
        >>> handler(1, 2)
        3
    """

    handlers: dict[
        Callable[..., Any], list[tuple[Pattern, Callable[..., Any]]]
    ] = {}
    graph: Any = None

    @classmethod
    def register(
        cls,
        fn: Callable[..., Any],
        parameters: tuple[str | Parameter, ...],
        handler: Callable[..., Any],
    ):
        """
        Registering function signature.

        Args:
            fn: The function to be registered.
            parameters: The parameters of the function to be registered.
            handler: The handler function.
        """
        _parameters = tuple(
            (
                Parameter.from_str(parameter)
                if isinstance(parameter, str)
                else parameter
            )
            for parameter in parameters
        )
        if fn not in cls.handlers:
            cls.handlers[fn] = []
        cls.handlers[fn].append((Pattern(*_parameters), handler))

    @classmethod
    def register_decorator(cls, fn: Callable[..., Any]):
        """
        Decorator mode of register, Used to register some complex functions.

        Args:
            fn: The function to be registered.

        Examples:
            >>> def builtin_add(a: int, b: int) -> int:
            ...     ...
            ...
            >>> @Dispatcher.register_decorator(builtin_add)
            ... def builtin_add_dispatcher(a: int, b: int) -> int:
            ...     return a + b
            ...
            >>> handler = Dispatcher.dispatch(builtin_add, 1, 2)
            >>> handler(1, 2)
            3
        """

        def decorator(handler: Callable[..., Any]):
            signature = inspect.signature(handler)
            parameters = tuple(
                Parameter.from_parameter(parameter)
                for parameter in signature.parameters.values()
            )
            cls.register(fn, parameters, handler)

        return decorator

    @classmethod
    def call(cls, fn, *args, **kwargs):
        func = cls.dispatch(fn, *args, **kwargs)
        if func is None:
            raise InnerError(
                f"Cannot find handler for {fn} with args {args} and kwargs {kwargs}"
            )
        return func(*args, **kwargs)

    @classmethod
    def dispatch(
        cls, fn: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Callable[..., Any] | None:
        """
        Find the matching handler from the registered functions.

        Args:
            fn: The function to be dispatched.
            args: The args of the function.
            kwargs: The kwargs of the function.
        """
        if not hashable(fn) or fn not in cls.handlers:
            return None
        for pattern, handler in cls.handlers[fn]:
            if pattern.match_inputs(*args, **kwargs):
                return handler
        return None
