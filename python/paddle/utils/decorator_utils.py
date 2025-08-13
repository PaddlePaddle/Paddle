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

import functools
import inspect
from collections.abc import Iterable
from typing import Any, Callable, TypeVar, cast

_F = TypeVar("_F", bound=Callable[..., Any])


class DecoratorBase:
    """Decorative base class, providing a universal decorative framework.

    Subclass only needs to implement the 'process' method to define the core logic.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs

    def __call__(self, func: _F) -> _F:
        """As an entry point for decorative applications"""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Pretreatment parameters
            processed_args, processed_kwargs = self.process(args, kwargs)
            return func(*processed_args, **processed_kwargs)

        wrapper.__signature__ = inspect.signature(func)
        return cast("_F", wrapper)

    def process(
        self, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """To be implemented by subclass"""
        raise NotImplementedError("Subclasses must implement this method")


# Example implementation: Parameter alias decorator
class ParamAliasDecorator(DecoratorBase):
    """Implementation of Decorator for Parameter Alias Processing"""

    def __init__(self, alias_mapping: dict[str, Iterable[str]]) -> None:
        super().__init__()
        # Check alias_mapping types
        if not isinstance(alias_mapping, dict):
            raise TypeError("alias_mapping must be a dictionary")
        for k, v in alias_mapping.items():
            if not isinstance(v, (list, tuple, set)):
                raise TypeError(f"Aliases for '{k}' must be iterable")

        # Build a reverse alias map for faster lookup
        self.alias_mapping = {}
        for original, aliases in alias_mapping.items():
            for alias in aliases:
                self.alias_mapping[alias] = original

    def process(
        self, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Process parameters to handle alias mapping"""
        if not kwargs:
            return args, kwargs

        processed_kwargs = kwargs
        alias_mapping = self.alias_mapping

        # Directly modify kwargs based on alias mapping (only modify if necessary)
        for alias, original in alias_mapping.items():
            if alias in processed_kwargs:
                if original not in processed_kwargs:
                    # Only modify the dictionary if necessary
                    processed_kwargs[original] = processed_kwargs.pop(alias)
                else:
                    raise ValueError(
                        f"Cannot specify both '{original}' and its alias '{alias}'"
                    )

        return args, processed_kwargs


def param_one_alias(alias_mapping):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not kwargs:
                return func(*args, **kwargs)
            if ("input" in kwargs) and ("x" not in kwargs):
                kwargs["x"] = kwargs.pop("input")
            return func(*args, **kwargs)

        wrapper.__signature__ = inspect.signature(func)
        return wrapper

    return decorator


# *size => shape decorator
class SizeArgsDecorator(DecoratorBase):
    """
    Usage Example:

    paddle.ones(1, dtype=paddle.float32)
    paddle.ones(1, 2, 3, dtype=paddle.float32)
    paddle.ones([1, 2, 3], dtype=paddle.float32)
    paddle.ones(size=[1, 2, 3], dtype=paddle.float32)

    paddle.ones([1, 2, 3], paddle.float32)
    paddle.ones(shape=[1, 2, 3], dtype=paddle.float32)
    """

    def process(
        self, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        if 'size' in kwargs:
            kwargs['shape'] = kwargs.pop('size')
        elif len(args) >= 1 and isinstance(args[0], int):
            kwargs['shape'] = list(args)
            args = ()

        return args, kwargs
