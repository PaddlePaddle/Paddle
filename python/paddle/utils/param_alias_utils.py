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
from typing import TYPE_CHECKING, Any, Callable, TypeVar, dict

if TYPE_CHECKING:
    from collections.abc import Iterable

F = TypeVar('F', bound=Callable[..., Any])


def param_alias(alias_mapping: dict[str, Iterable[str]]) -> Callable[[F], F]:
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

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not kwargs:
                return func(*args, **kwargs)
            processed_kwargs = kwargs.copy()
            for original, aliases in alias_mapping.items():
                for alias in aliases:
                    if alias in processed_kwargs:
                        if original not in processed_kwargs:
                            processed_kwargs[original] = processed_kwargs.pop(
                                alias
                            )
                        else:
                            raise ValueError(
                                f"Cannot specify both '{original}' and its alias '{alias}'"
                            )
            return func(*args, **processed_kwargs)

        return wrapper  # type: ignore

    return decorator
