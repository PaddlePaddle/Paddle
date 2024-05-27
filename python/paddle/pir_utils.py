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


from functools import wraps

import paddle
from paddle.framework.dtype import bind_datatype, bind_vartype


def _switch_to_pir_():
    paddle.base.framework.global_var._use_pir_api_ = True
    paddle.framework.set_flags({"FLAGS_enable_pir_in_executor": True})
    paddle.pir.register_paddle_dialect()
    # TODO find a better place to init the registion of dist dialect.
    paddle.pir.register_dist_dialect()

    paddle.base.Program = paddle.pir.Program
    paddle.base.program_guard = paddle.pir.core.program_guard
    paddle.base.default_main_program = paddle.pir.core.default_main_program
    paddle.base.default_startup_program = (
        paddle.pir.core.default_startup_program
    )
    paddle.static.Program = paddle.pir.Program
    paddle.static.program_guard = paddle.pir.core.program_guard
    paddle.static.default_main_program = paddle.pir.core.default_main_program
    paddle.static.default_startup_program = (
        paddle.pir.core.default_startup_program
    )


def _switch_to_old_ir_():
    paddle.base.framework.global_var._use_pir_api_ = False
    paddle.framework.set_flags({"FLAGS_enable_pir_in_executor": False})

    paddle.base.Program = paddle.base.framework.Program
    paddle.base.program_guard = paddle.base.framework.program_guard
    paddle.base.default_main_program = (
        paddle.base.framework.default_main_program
    )
    paddle.base.default_startup_program = (
        paddle.base.framework.default_startup_program
    )
    paddle.static.Program = paddle.base.framework.Program
    paddle.static.program_guard = paddle.base.framework.program_guard
    paddle.static.default_main_program = (
        paddle.base.framework.default_main_program
    )
    paddle.static.default_startup_program = (
        paddle.base.framework.default_startup_program
    )


class IrGuard:
    def __enter__(self):
        self.in_dygraph_outside = paddle.base.framework.in_dygraph_mode()
        self.old_flag = paddle.base.framework.get_flags("FLAGS_enable_pir_api")[
            "FLAGS_enable_pir_api"
        ]
        if self.in_dygraph_outside:
            paddle.enable_static()
        if not self.old_flag:
            paddle.framework.set_flags({"FLAGS_enable_pir_api": True})
            paddle.base.framework.global_var._use_pir_api_ = True
            bind_datatype()
            self._switch_to_pir()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.in_dygraph_outside:
            paddle.disable_static()
        if not self.old_flag:
            paddle.framework.set_flags({"FLAGS_enable_pir_api": False})
            paddle.base.framework.global_var._use_pir_api_ = False
            bind_vartype()
            self._switch_to_old_ir()

    def _switch_to_pir(self):
        if paddle.base.framework.get_flags("FLAGS_enable_pir_api")[
            "FLAGS_enable_pir_api"
        ]:
            _switch_to_pir_()

    def _switch_to_old_ir(self):
        if not paddle.base.framework.get_flags("FLAGS_enable_pir_api")[
            "FLAGS_enable_pir_api"
        ]:
            _switch_to_old_ir_()
        else:
            raise RuntimeError(
                "IrGuard._switch_to_old_ir only work when paddle.framework.in_pir_mode() is false, \
                please set FLAGS_enable_pir_api = false"
            )


class OldIrGuard:
    def __enter__(self):
        self.in_dygraph_outside = paddle.base.framework.in_dygraph_mode()
        self.old_flag = paddle.base.framework.get_flags("FLAGS_enable_pir_api")[
            "FLAGS_enable_pir_api"
        ]
        if self.in_dygraph_outside:
            paddle.enable_static()
        if self.old_flag:
            paddle.framework.set_flags({"FLAGS_enable_pir_api": False})
            paddle.base.framework.global_var._use_pir_api_ = False
            bind_vartype()
            _switch_to_old_ir_()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.in_dygraph_outside:
            paddle.disable_static()
        if self.old_flag:
            paddle.framework.set_flags({"FLAGS_enable_pir_api": True})
            paddle.base.framework.global_var._use_pir_api_ = True
            bind_datatype()
            _switch_to_pir_()


class DygraphPirGuard:
    def __enter__(self):
        self.old_flag = paddle.base.framework.get_flags("FLAGS_enable_pir_api")[
            "FLAGS_enable_pir_api"
        ]
        if not self.old_flag:
            paddle.framework.set_flags({"FLAGS_enable_pir_api": True})
            paddle.base.framework.global_var._use_pir_api_ = True
            bind_datatype()
            self._switch_to_pir()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.old_flag:
            paddle.framework.set_flags({"FLAGS_enable_pir_api": False})
            paddle.base.framework.global_var._use_pir_api_ = False
            bind_vartype()
            self._switch_to_old_ir()

    def _switch_to_pir(self):
        if paddle.base.framework.get_flags("FLAGS_enable_pir_api")[
            "FLAGS_enable_pir_api"
        ]:
            _switch_to_pir_()

    def _switch_to_old_ir(self):
        if not paddle.base.framework.get_flags("FLAGS_enable_pir_api")[
            "FLAGS_enable_pir_api"
        ]:
            _switch_to_old_ir_()
        else:
            raise RuntimeError(
                "IrGuard._switch_to_old_ir only work when paddle.framework.in_pir_mode() is false, \
                please set FLAGS_enable_pir_api = false"
            )


class DygraphOldIrGuard:
    def __enter__(self):
        self.old_flag = paddle.base.framework.get_flags("FLAGS_enable_pir_api")[
            "FLAGS_enable_pir_api"
        ]
        if self.old_flag:
            paddle.framework.set_flags({"FLAGS_enable_pir_api": False})
            paddle.base.framework.global_var._use_pir_api_ = False
            bind_vartype()
            _switch_to_old_ir_()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.old_flag:
            paddle.framework.set_flags({"FLAGS_enable_pir_api": True})
            paddle.base.framework.global_var._use_pir_api_ = True
            bind_datatype()
            _switch_to_pir_()


def test_with_pir_api(func):
    @wraps(func)
    def impl(*args, **kwargs):
        with OldIrGuard():
            func(*args, **kwargs)
        with IrGuard():
            func(*args, **kwargs)

    return impl


def test_with_old_ir_only(func):
    @wraps(func)
    def impl(*args, **kwargs):
        with OldIrGuard():
            func(*args, **kwargs)

    return impl


def test_with_dygraph_pir(func):
    @wraps(func)
    def impl(*args, **kwargs):
        with DygraphOldIrGuard():
            func(*args, **kwargs)

        with DygraphPirGuard():
            func(*args, **kwargs)

    return impl
