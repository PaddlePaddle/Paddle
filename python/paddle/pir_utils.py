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


class IrGuard:
    def __init__(self):
        self.in_dygraph_outside = False

    def __enter__(self):
        self.in_dygraph_outside = paddle.base.framework.in_dygraph_mode()
        if self.in_dygraph_outside:
            paddle.enable_static()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.in_dygraph_outside:
            paddle.disable_static()

    def _switch_to_pir(self):
        paddle.base.framework.global_var._use_pir_api_ = (
            paddle.base.framework.get_flags("FLAGS_enable_pir_api")[
                "FLAGS_enable_pir_api"
            ]
        )
        if paddle.base.framework.global_var._use_pir_api_:
            paddle.framework.set_flags({"FLAGS_enable_pir_in_executor": True})
            paddle.pir.register_paddle_dialect()
            # TODO find a better place to init the registion of dist dialect.
            paddle.pir.register_dist_dialect()

            paddle.base.Program = paddle.pir.Program
            paddle.base.program_guard = paddle.pir.core.program_guard
            paddle.static.Program = paddle.pir.Program
            paddle.static.program_guard = paddle.pir.core.program_guard
            paddle.static.default_main_program = (
                paddle.pir.core.default_main_program
            )
            paddle.static.default_startup_program = (
                paddle.pir.core.default_startup_program
            )


def test_with_pir_api(func):
    @wraps(func)
    def impl(*args, **kwargs):
        func(*args, **kwargs)
        with IrGuard():
            func(*args, **kwargs)

    return impl
