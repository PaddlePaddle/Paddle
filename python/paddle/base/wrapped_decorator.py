# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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


import contextlib
import functools
import inspect
from typing import Callable, TypeVar

from typing_extensions import ParamSpec

_InputT = ParamSpec("_InputT")
_RetT1 = TypeVar("_RetT1")
_RetT2 = TypeVar("_RetT2")

__all__ = []


def copy_signature(
    src: Callable[_InputT, _RetT1], dst: Callable[_InputT, _RetT1]
):
    src_signature = inspect.signature(src)
    dst.__signature__ = src_signature.replace(
        parameters=list(src_signature.parameters.values())
    )


def wrap_decorator(
    decorator_func: Callable[
        [Callable[_InputT, _RetT1]], Callable[_InputT, _RetT2]
    ],
) -> Callable[[Callable[_InputT, _RetT1]], Callable[_InputT, _RetT2]]:
    @functools.wraps(decorator_func)
    def __impl__(func: Callable[_InputT, _RetT1]) -> Callable[_InputT, _RetT2]:
        decorated = decorator_func(func)

        @functools.wraps(func)
        def wrapper(*args: _InputT.args, **kwargs: _InputT.kwargs) -> _RetT2:
            return decorated(*args, **kwargs)

        copy_signature(func, wrapper)
        return wrapper

    return __impl__


signature_safe_contextmanager = wrap_decorator(contextlib.contextmanager)
