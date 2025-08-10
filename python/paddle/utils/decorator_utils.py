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

import functools
import inspect
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    TypeVar,
)

from typing_extensions import ParamSpec

if TYPE_CHECKING:
    from collections.abc import Iterable


_P = ParamSpec("_P")
_R = TypeVar("_R")
_DecoratedFunc = Callable[_P, _R]


class DecoratorBase(Generic[_P, _R]):
    """Base class for creating decorators"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs

    def __call__(self, func: Callable[..., _R]) -> Callable[..., _R]:
        """Wrap the function with the decorator"""

        @functools.wraps(func)
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            # Call the core logic for parameter processing
            processed_args, processed_kwargs = self.process(args, kwargs)
            return func(*processed_args, **processed_kwargs)

        wrapper.__signature__ = inspect.signature(func)
        return wrapper

    def process(
        self, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """To be implemented by subclass"""
        raise NotImplementedError("Subclasses must implement this method")


class ParamAliasDecorator(DecoratorBase[_P, _R]):
    """Decorator for parameter alias processing"""

    def __init__(self, alias_mapping: dict[str, Iterable[str]]) -> None:
        super().__init__()
        # Check alias_mapping types
        # if not isinstance(alias_mapping, dict):
        #     raise TypeError("alias_mapping must be a dictionary")
        # for k, v in alias_mapping.items():
        #     if not isinstance(v, (list, tuple, set)):
        #         raise TypeError(f"Aliases for '{k}' must be iterable")

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
