# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved
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

import paddle


def _switch_to_new_ir():
    if paddle.ir.core._use_new_ir_api():
        paddle.framework.set_flags({"FLAGS_enable_new_ir_in_executor": True})
        paddle.ir.register_paddle_dialect()
        paddle.static.Program = paddle.ir.Program
        paddle.fluid.Program = paddle.ir.Program
        paddle.fluid.program_guard = paddle.ir.core.program_guard
        paddle.static.program_guard = paddle.ir.core.program_guard
        paddle.framework.default_main_program = (
            paddle.ir.core.default_main_program
        )
