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

from paddle.cinn import ir

from .ir_context import ForContext


# Python's range() function calls the sequential()
def sequential(min, extent=None):
    if extent is None:
        extent = min
        min = ir.Expr(0)
    if not isinstance(min, ir.Expr):
        min = ir.Expr(min)
    if not isinstance(extent, ir.Expr):
        extent = ir.Expr(extent)
    return ForContext(min, extent)
