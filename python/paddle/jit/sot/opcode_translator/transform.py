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
from functools import partial
from typing import TYPE_CHECKING

from ..utils import CodeStatus, EventGuard, log, log_do
from .executor.opcode_executor import CustomCode, InstructionTranslatorCache
from .skip_files import need_skip

if TYPE_CHECKING:
    pass


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
        # is generator
        if frame.f_code.co_flags & 0x20 > 0:
            return CustomCode(None, True)

        if need_skip(frame):
            return CustomCode(None, False)

        log(
            2,
            "[eval_frame_callback] start to translate: "
            + str(frame.f_code)
            + "\n",
        )
        log_do(4, partial(print_locals, frame))

        log(3, f"[transform] OriginCode: {frame.f_code.co_name}\n")
        log_do(3, lambda: dis.dis(frame.f_code))

        custom_code = InstructionTranslatorCache()(frame, **kwargs)

        if custom_code.code is None:
            log(
                3,
                "[transform] NewCode (same as origin code): "
                + frame.f_code.co_name
                + "\n",
            )
            used_code = frame.f_code
        else:
            log(
                3,
                "[transform] NewCode: " + custom_code.code.co_name + "\n",
            )
            log_do(3, lambda: dis.dis(custom_code.code))
            used_code = custom_code.code

        # just check those codes which need open eval_frame
        if custom_code.disable_eval_frame is False and CodeStatus().check_code(
            used_code
        ):
            log(
                3,
                "[transform] Code has found no graph, block it.",
            )
            return CustomCode(None, True)

        return custom_code
