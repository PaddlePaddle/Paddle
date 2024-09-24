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

import dis
from typing import TYPE_CHECKING

from ..profiler import EventGuard
from ..utils import log_do, log_enabled, log_format
from .executor.executor_cache import OpcodeExecutorCache

if TYPE_CHECKING:
    from .custom_code import CustomCode


def print_locals(frame):
    local_key = [
        key for key in frame.f_locals.keys() if not key.startswith("__")
    ]
    print(
        f"[eval_frame_callback] {frame.f_code.co_name} with locals {local_key}"
    )
    print(
        f"[eval_frame_callback] {' ' * len(frame.f_code.co_name)} with cellvars + freevars:  {frame.f_code.co_cellvars + frame.f_code.co_freevars}"
    )

    def convert_obj(obj):
        import paddle

        if isinstance(obj, paddle.Tensor):
            return "Tensor(" + str(obj.shape) + ")"
        if isinstance(obj, list):
            return [convert_obj(i) for i in obj]
        return obj

    for key in local_key:
        print(
            f"[eval_frame_callback] {' ' * len(frame.f_code.co_name)} {key} = {convert_obj(frame.f_locals[key])}"
        )


def eval_frame_callback(frame, **kwargs) -> CustomCode:
    with EventGuard(
        f"eval_frame_callback: {frame.f_code.co_name}", event_level=2
    ):
        # A quick way to check if the log is enabled
        need_log = log_enabled(1)
        if need_log:
            log_format(
                2,
                "[eval_frame_callback] start to translate: {}\n",
                frame.f_code,
            )
            log_do(4, lambda: print_locals(frame))

            log_format(
                3,
                "[eval_frame_callback] OriginCode: {}\n",
                frame.f_code.co_name,
            )
            log_do(3, lambda: dis.dis(frame.f_code))

        custom_code = OpcodeExecutorCache()(frame, **kwargs)

        if need_log:
            if custom_code.code is None:
                log_format(
                    3,
                    "[eval_frame_callback] NewCode (same as origin code): {}\n",
                    frame.f_code.co_name,
                )
            else:
                log_format(
                    3,
                    "[eval_frame_callback] NewCode: {}\n",
                    custom_code.code.co_name,
                )
                log_do(3, lambda: dis.dis(custom_code.code))

        return custom_code
