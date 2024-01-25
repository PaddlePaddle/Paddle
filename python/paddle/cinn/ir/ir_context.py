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

from paddle.base import core
from paddle.cinn import ir


# Encapsulated cinn::pybind::IRBuilder in C++
class IRBuilder:
    def __init__(self):
        self.ir_builder = core.cinn.ir.IRBuilder()

    def __enter__(self):
        self.ir_builder.EnterWithContext()
        return self

    def __exit__(self, ptype, value, trace) -> None:
        if ptype is None and value is None:
            self.ir_builder.ExitWithContext()

    def get(self):
        return self.ir_builder.get_result()


# Encapsulated cinn::pybind::IRContext in C++
class IRContext:
    def __init__(self, ir_ctx):
        self.ir_ctx = ir_ctx

    def __enter__(self):
        self.ir_ctx.EnterWithContext()

    def __exit__(self, ptype, value, trace) -> None:
        if ptype is None and value is None:
            self.ir_ctx.ExitWithContext()


# Encapsulated cinn::pybind::ScheduleBlockContext in C++
class ScheduleBlockContext(IRContext):
    def __init__(self, name):
        self.ir_ctx = core.cinn.ir.IRContext.MakeScheduleBlockContext(name)


# Encapsulated cinn::pybind::LowerFuncContext in C++
class LowerFuncContext(IRContext):
    def __init__(self, name):
        self.ir_ctx = core.cinn.ir.IRContext.MakeLowerFunctionContext(name)


# Encapsulated cinn::pybind::ForContext in C++
class ForContext(IRContext):
    def __init__(self, min, extent):
        self.ir_ctx = ir.Sequential(min, extent)

    def __enter__(self):
        super().__enter__()
        return self.ir_ctx.get_for_loop_var()


# Encapsulated cinn::pybind::IfContext in C++
class IfContext(IRContext):
    def __init__(self, expr):
        self.ir_ctx = core.cinn.ir.IRContext.MakeIfContext(expr)


# Encapsulated cinn::pybind::ThenContext in C++
class ThenContext(IRContext):
    def __init__(self):
        self.ir_ctx = core.cinn.ir.IRContext.MakeThenContext()


# Encapsulated cinn::pybind::ElseContext in C++
class ElseContext(IRContext):
    def __init__(self):
        self.ir_ctx = core.cinn.ir.IRContext.MakeElseContext()
