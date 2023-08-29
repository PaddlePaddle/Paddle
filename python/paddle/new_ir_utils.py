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

from .fluid.wrapped_decorator import signature_safe_contextmanager


class IrChange:
    def __init__(self):
        if not paddle.ir.core._use_new_ir_api():
            self.old_Program = paddle.static.Program
            self.old_program_guard = paddle.fluid.program_guard
            self.old_default_main_program = paddle.static.default_main_program
        else:
            raise RuntimeError(
                "IrChange only init when paddle.ir.core._use_new_ir_api() is false, \
                please set FLAGS_enable_new_ir_api = false"
            )

    def _switch_to_new_ir(self):
        if paddle.ir.core._use_new_ir_api():
            paddle.framework.set_flags(
                {"FLAGS_enable_new_ir_in_executor": True}
            )
            paddle.ir.register_paddle_dialect()
            paddle.static.Program = paddle.ir.Program
            paddle.fluid.Program = paddle.ir.Program
            paddle.fluid.program_guard = paddle.ir.core.program_guard
            paddle.static.program_guard = paddle.ir.core.program_guard
            paddle.framework.default_main_program = (
                paddle.ir.core.default_main_program
            )

    def _switch_to_old_ir(self):
        if not paddle.ir.core._use_new_ir_api():
            paddle.framework.set_flags(
                {"FLAGS_enable_new_ir_in_executor": False}
            )
            paddle.static.Program = self.old_Program
            paddle.fluid.Program = self.old_Program
            paddle.fluid.program_guard = self.old_program_guard
            paddle.static.program_guard = self.old_program_guard
            paddle.framework.default_main_program = (
                self.old_default_main_program
            )
        else:
            raise RuntimeError(
                "IrChange._switch_to_old_ir only work when paddle.ir.core._use_new_ir_api() is false, \
                please set FLAGS_enable_new_ir_api = false"
            )


@signature_safe_contextmanager
def _newir_guard():
    ir_change = IrChange()
    paddle.framework.set_flags({"FLAGS_enable_new_ir_api": True})
    ir_change._switch_to_new_ir()
    try:
        yield
    finally:
        paddle.framework.set_flags({"FLAGS_enable_new_ir_api": False})
        ir_change._switch_to_old_ir()
