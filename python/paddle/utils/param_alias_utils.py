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
from typing import Callable, dict, list


def param_alias(alias_mapping: dict[str, list[str]]) -> Callable:
    """
    Decorator to map parameter names between different API conventions.
    """

    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not kwargs:
                return func(*args, **kwargs)
            for original, aliases in alias_mapping.items():
                for alias in aliases:
                    if alias in kwargs:
                        if original not in kwargs:
                            kwargs[original] = kwargs[alias]
                        else:
                            raise KeyError(
                                f"Both {original} and {alias} are provided. Please specify only one."
                            )
                        del kwargs[alias]
            return func(*args, **kwargs)

        return wrapper

    return decorator
