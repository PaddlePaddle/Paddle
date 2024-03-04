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

import os
from contextlib import contextmanager
from functools import wraps

from paddle.framework import core

_event_level = int(os.environ.get("EVENT_LEVEL", "0"))


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
    try:
        global _event_level
        need_pop = False
        if _event_level >= event_level:
            core.nvprof_nvtx_push(event_name)
            need_pop = True
        yield
    finally:
        if need_pop:
            core.nvprof_nvtx_pop()


def event_register(event_name, event_level=1):
    def event_wrapper(func):
        @wraps(func)
        def call_with_event(*args, **kwargs):
            with EventGuard(event_name, event_level=event_level):
                return func(*args, **kwargs)

        return call_with_event

    def do_nothing(func):
        return func

    global _event_level
    if _event_level >= event_level:
        return event_wrapper
    else:
        return do_nothing
