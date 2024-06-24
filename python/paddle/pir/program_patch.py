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

from paddle.base.wrapped_decorator import signature_safe_contextmanager

from . import Program

_already_patch_program = False


def monkey_patch_program():
    @signature_safe_contextmanager
    def _lr_schedule_guard(self, is_with_opt=False):
        # TODO(dev): Currently there has not equivalent of op_role in PIR
        # mode, so we simply remove _lr_schedule_guard here, this should
        # be fixed in the future.
        yield

    program_attrs = {
        "_lr_schedule_guard": _lr_schedule_guard,
    }

    global _already_patch_program
    if not _already_patch_program:
        for attr, value in program_attrs.items():
            setattr(Program, attr, value)

        _already_patch_program = True
