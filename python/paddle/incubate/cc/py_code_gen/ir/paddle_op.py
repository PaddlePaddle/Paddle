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

from dataclasses import dataclass

from paddle.incubate.cc.py_code_gen.ir import ir_op
from paddle.incubate.cc.py_code_gen.ir_converters.paddle_tensor_converter import (
    ConvertToPaddleTensor,
)


@dataclass
class Op(ir_op.Op):
    base_op: ir_op.Op = None

    def GetResult(self, i):
        return ConvertToPaddleTensor(ir_op.Op.GetResult(self, i))
