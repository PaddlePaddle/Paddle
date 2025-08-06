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
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Callable, TypeVar

from typing_extensions import ParamSpec

from paddle.base.wrapped_decorator import signature_safe_contextmanager

if TYPE_CHECKING:
    from collections.abc import Iterable


_InputT = ParamSpec("_InputT")
_RetT = TypeVar("_RetT")


def param_alias(
    alias_mapping: dict[str, Iterable[str]],
) -> Callable[[Callable[_InputT, _RetT]], Callable[_InputT, _RetT]]:
    """Decorator for handling parameter aliases in function calls.

    Args:
        alias_mapping: Dictionary mapping original parameter names to their aliases.
                      Example: {'original': ['alias1', 'alias2']}

    Returns:
        A decorator that processes parameter aliases before calling the original function.
    """
    if not isinstance(alias_mapping, dict):
        raise TypeError("alias_mapping must be a dictionary")
    for k, v in alias_mapping.items():
        if not isinstance(v, (list, tuple, set)):
            raise TypeError(f"Aliases for '{k}' must be iterable")

    @signature_safe_contextmanager
    def alias_context(
        *args: _InputT.args, **kwargs: _InputT.kwargs
    ) -> Iterable[dict[str, Any]]:
        processed_kwargs = kwargs.copy()
        for original, aliases in alias_mapping.items():
            for alias in aliases:
                if alias in processed_kwargs:
                    if original not in processed_kwargs:
                        processed_kwargs[original] = processed_kwargs.pop(alias)
                    else:
                        raise ValueError(
                            f"Cannot specify both '{original}' and its alias '{alias}'"
                        )
        yield processed_kwargs

    def decorator(func: Callable[_InputT, _RetT]) -> Callable[_InputT, _RetT]:
        @functools.wraps(func)
        def wrapper(*args: _InputT.args, **kwargs: _InputT.kwargs) -> _RetT:
            if not kwargs:
                return func(*args, **kwargs)
            with alias_context(*args, **kwargs) as processed_kwargs:
                return func(*args, **processed_kwargs)

        return wrapper

    return decorator
