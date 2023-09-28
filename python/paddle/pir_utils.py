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


class IrGuard:
    def __init__(self):
        self.in_dygraph_outside = False
        old_flag = paddle.base.framework.get_flags("FLAGS_enable_pir_api")
        paddle.base.framework.set_flags({"FLAGS_enable_pir_api": False})
        paddle.base.framework.global_var._use_pir_api_ = False
        if not paddle.base.framework.get_flags("FLAGS_enable_pir_api")[
            "FLAGS_enable_pir_api"
        ]:
            self.old_Program = paddle.static.Program
            self.old_program_guard = paddle.base.program_guard
            self.old_default_main_program = paddle.static.default_main_program
            self.old_default_startup_program = (
                paddle.static.default_startup_program
            )
        else:
            raise RuntimeError(
                "IrGuard only init when paddle.framework.in_pir_mode(): is false, \
                please set FLAGS_enable_pir_api = false"
            )
        paddle.base.framework.set_flags(old_flag)
        paddle.base.framework.global_var._use_pir_api_ = old_flag[
            "FLAGS_enable_pir_api"
        ]

    def __enter__(self):
        self.in_dygraph_outside = paddle.base.framework.in_dygraph_mode()
        if self.in_dygraph_outside:
            paddle.enable_static()
        paddle.framework.set_flags({"FLAGS_enable_pir_api": True})
        paddle.base.framework.global_var._use_pir_api_ = True
        self._switch_to_pir()

    def __exit__(self, exc_type, exc_val, exc_tb):
        paddle.framework.set_flags({"FLAGS_enable_pir_api": False})
        paddle.base.framework.global_var._use_pir_api_ = False
        self._switch_to_old_ir()
        if self.in_dygraph_outside:
            paddle.disable_static()

    def _switch_to_pir(self):
        if paddle.base.framework.get_flags("FLAGS_enable_pir_api")[
            "FLAGS_enable_pir_api"
        ]:
            paddle.framework.set_flags(
                {"FLAGS_enable_new_ir_in_executor": True}
            )
            paddle.pir.register_paddle_dialect()
            paddle.static.Program = paddle.pir.Program
            paddle.base.Program = paddle.pir.Program
            paddle.base.program_guard = paddle.pir.core.program_guard
            paddle.static.program_guard = paddle.pir.core.program_guard
            paddle.static.default_main_program = (
                paddle.pir.core.default_main_program
            )
            paddle.static.default_startup_program = (
                paddle.pir.core.default_startup_program
            )

    def _switch_to_old_ir(self):
        if not paddle.base.framework.get_flags("FLAGS_enable_pir_api")[
            "FLAGS_enable_pir_api"
        ]:
            paddle.framework.set_flags(
                {"FLAGS_enable_new_ir_in_executor": False}
            )
            paddle.static.Program = self.old_Program
            paddle.base.Program = self.old_Program
            paddle.base.program_guard = self.old_program_guard
            paddle.static.program_guard = self.old_program_guard
            paddle.static.default_main_program = self.old_default_main_program
            paddle.static.default_startup_program = (
                self.old_default_startup_program
            )
        else:
            raise RuntimeError(
                "IrGuard._switch_to_old_ir only work when paddle.framework.in_pir_mode() is false, \
                please set FLAGS_enable_pir_api = false"
            )
