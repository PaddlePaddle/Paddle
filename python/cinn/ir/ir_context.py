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

from cinn import ir

from .. import core_api


class IRBuilder:
    def __init__(self):
        self.ir_builder = core_api.ir.IRBuilder()

    def __enter__(self):
        self.ir_builder.EnterWithContext()
        return self

    def __exit__(
        self, ptype, value, trace
    ) -> None:  # pylint: disable=unused-argument
        if ptype is None and value is None:
            self.ir_builder.ExitWithContext()

    def get(self):
        return self.ir_builder.get_result()


class IRContext:
    def __init__(self, ir_ctx):
        self.ir_ctx = ir_ctx

    def __enter__(self):
        self.ir_ctx.EnterWithContext()

    def __exit__(self, ptype, value, trace) -> None:
        if ptype is None and value is None:
            self.ir_ctx.ExitWithContext()


class ScheduleBlockContext(IRContext):
    def __init__(self, name):
        self.ir_ctx = core_api.ir.IRContext.MakeScheduleBlockContext(name)


class LowerFuncContext(IRContext):
    def __init__(self, name):
        self.ir_ctx = core_api.ir.IRContext.MakeLowerFunctionContext(name)


class ForContext(IRContext):
    def __init__(self, min, extent):
        self.ir_ctx = ir.Sequential(min, extent)

    def __enter__(self):
        super().__enter__()
        return self.ir_ctx.get_for_loop_var()


class IfContext(IRContext):
    def __init__(self, expr):
        self.ir_ctx = core_api.ir.IRContext.MakeIfContext(expr)


class ThenContext(IRContext):
    def __init__(self):
        self.ir_ctx = core_api.ir.IRContext.MakeThenContext()


class ElseContext(IRContext):
    def __init__(self):
        self.ir_ctx = core_api.ir.IRContext.MakeElseContext()
