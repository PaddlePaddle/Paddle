# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from paddle.incubate.cc.py_code_gen.ir.ir_symbol import ShapeOrDataDimExprs


@dataclass
class Tensor:
    local_name_prefix: str
    name: str
    arg_name_as_input: str | None
    defining_op_name: str | None
    type: Type  # noqa: F821
    dim_exprs: ShapeOrDataDimExprs

    @property
    def shape(self):
        return self.type.shape

    @property
    def dtype(self):
        return self.type.dtype
