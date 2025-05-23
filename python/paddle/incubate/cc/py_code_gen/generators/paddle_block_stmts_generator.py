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

from typing import TYPE_CHECKING

from paddle.incubate.cc.py_code_gen.generators.paddle_func_body_generator import (
    PaddleFuncBodyGenerator,
)

if TYPE_CHECKING:
    from paddle.incubate.cc.py_code_gen.generators.block_name_generator import (
        BlockNameGenerator,
    )
    from paddle.incubate.cc.py_code_gen.ir.ir_block import Block
    from paddle.incubate.cc.py_code_gen.ir.ir_tensor import Tensor


class PaddleBlockStmtsGenerator:

    def __init__(
        self,
        op_name2custom_op_call_generator,
        block_name_generator: BlockNameGenerator,
    ):
        self.op_name2custom_op_call_generator = op_name2custom_op_call_generator
        self.block_name_generator = block_name_generator

    def Generate(
        self, block: Block
    ) -> tuple[list[Tensor], list[PyCodeStmt]]:  # noqa: F821
        paddle_func_body_generator = PaddleFuncBodyGenerator(
            block.block_func,
            self.op_name2custom_op_call_generator,
            self.block_name_generator,
        )
        return paddle_func_body_generator.Generate(block.free_vars, block.args)
