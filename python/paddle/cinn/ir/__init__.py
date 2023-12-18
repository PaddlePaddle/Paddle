# Copyright (c) 2021 CINN Authors. All Rights Reserved.
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
from .ir_context import (  # noqa: F401
    ElseContext,
    ForContext,
    IfContext,
    IRBuilder,
    IRContext,
    LowerFuncContext,
    ScheduleBlockContext,
    ThenContext,
)

from .ir_api import sequential

__all__ = []
ignore_cpp_module = [
    "ElseContext",
    "ForContext",
    "IfContext",
    "IRBuilder",
    "IRContext",
    "ForContext",
    "IRContext",
    "LowerFuncContext",
    "ScheduleBlockContext",
    "ThenContext",
]

for name in dir(core.cinn.ir):
    if name not in ignore_cpp_module:
        globals()[name] = getattr(core.cinn.ir, name)
        __all__.append(name)
