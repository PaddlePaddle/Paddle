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
from typing import Any, Callable, TypeVar, cast

F = TypeVar('F', bound=Callable[..., Any])


def forbid_keywords(
    illegal_keys: list[str] | str, correct_func_name: str
) -> Callable[[F], F]:
    """
    A decorator that hints users to use the correct `compat` functions, when erroneous keyword arguments are detected

    Args:
        illegal_keys: list[str] | str - Forbidden keyword names
        correct_func_name: str - Recommended function name
    """
    keys = [illegal_keys] if isinstance(illegal_keys, str) else illegal_keys

    def decorator(func: F) -> F:
        orig_sig = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            found_keys = [key for key in illegal_keys if key in kwargs]

            if found_keys:
                keys_str = ", ".join(f"'{key}'" for key in found_keys)
                plural = "s" if len(found_keys) > 1 else ""

                raise TypeError(
                    f"{func.__name__}() received unexpected keyword argument{plural} {keys_str}. "
                    f"\nDid you mean to use {correct_func_name}() instead?"
                )

            return func(*args, **kwargs)

        # Important: function signatures / specs should be copied to avoid erroneous input/output extraction (particularly in static graph, like test_split_op.py)
        wrapper.__signature__ = orig_sig
        if hasattr(func, "__defaults__"):
            wrapper.__defaults__ = func.__defaults__
        if hasattr(func, "__kwdefaults__"):
            wrapper.__kwdefaults__ = func.__kwdefaults__
        wrapper.__wrapped__ = func

        return cast('F', wrapper)

    return decorator
