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

from contextlib import contextmanager
from functools import wraps
from typing import Callable, TypeVar

from typing_extensions import ParamSpec

from paddle.framework import core

from .dy2static.utils import ENV_SOT_EVENT_LEVEL

P = ParamSpec("P")
T = TypeVar("T")


class SotProfiler:
    def __enter__(self):
        self.enable()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disable()

    def enable(self, tag=None):
        core.nvprof_start()
        core.nvprof_enable_record_event()

    def disable(self):
        core.nvprof_stop()


@contextmanager
def EventGuard(event_name, event_level=1):
    need_pop = False
    try:
        if ENV_SOT_EVENT_LEVEL.get() >= event_level:
            core.nvprof_nvtx_push(event_name)
            need_pop = True
        yield
    finally:
        if need_pop:
            core.nvprof_nvtx_pop()


def event_register(
    event_name_formatter: Callable[P, str] | str, event_level=1
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    def event_wrapper(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def call_with_event(*args: P.args, **kwargs: P.kwargs):
            event_name = (
                event_name_formatter(*args, **kwargs)
                if callable(event_name_formatter)
                else event_name_formatter
            )
            with EventGuard(event_name, event_level=event_level):
                return func(*args, **kwargs)

        return call_with_event

    def do_nothing(func: Callable[P, T]) -> Callable[P, T]:
        return func

    if ENV_SOT_EVENT_LEVEL.get() >= event_level:
        return event_wrapper
    else:
        return do_nothing
