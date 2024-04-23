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

import builtins
import types
from typing import Callable, TypeVar

from typing_extensions import ParamSpec

T = TypeVar("T")
P = ParamSpec("P")

NO_BREAKGRAPH_CODES: set[types.CodeType] = set()
NO_FALLBACK_CODES: set[types.CodeType] = set()


def assert_true(input: bool):
    assert input


def print(*args, **kwargs):
    builtins.print("[Dygraph]", *args, **kwargs)


def breakpoint():
    import paddle

    old = paddle.framework.core.set_eval_frame(None)
    builtins.breakpoint()  # noqa: T100
    paddle.framework.core.set_eval_frame(old)


def check_no_breakgraph(fn: Callable[P, T]) -> Callable[P, T]:
    NO_BREAKGRAPH_CODES.add(fn.__code__)
    return fn


def breakgraph():
    pass


def check_no_fallback(fn: Callable[P, T]) -> Callable[P, T]:
    NO_FALLBACK_CODES.add(fn.__code__)
    return fn


def fallback():
    pass


def in_sot():
    return False
