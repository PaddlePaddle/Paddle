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
from paddle.framework.dtype import bind_datatype, bind_vartype


def _switch_to_pir_():
    bind_datatype()
    paddle.base.framework.global_var._use_pir_api_ = True
    paddle.framework.set_flags({"FLAGS_enable_pir_in_executor": True})
    paddle.pir.register_paddle_dialect()
    # TODO find a better place to init the registration of dist dialect.
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
    bind_vartype()
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
            self._switch_to_pir()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.in_dygraph_outside:
            paddle.disable_static()
        if not self.old_flag:
            paddle.framework.set_flags({"FLAGS_enable_pir_api": False})
            paddle.base.framework.global_var._use_pir_api_ = False
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
            _switch_to_old_ir_()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.in_dygraph_outside:
            paddle.disable_static()
        if self.old_flag:
            paddle.framework.set_flags({"FLAGS_enable_pir_api": True})
            _switch_to_pir_()


class DygraphPirGuard:
    def __enter__(self):
        self.old_flag = paddle.base.framework.get_flags("FLAGS_enable_pir_api")[
            "FLAGS_enable_pir_api"
        ]
        if not self.old_flag:
            paddle.framework.set_flags({"FLAGS_enable_pir_api": True})
            self._switch_to_pir()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.old_flag:
            paddle.framework.set_flags({"FLAGS_enable_pir_api": False})
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
            _switch_to_old_ir_()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.old_flag:
            paddle.framework.set_flags({"FLAGS_enable_pir_api": True})
            _switch_to_pir_()


def get_memory(value):
    from paddle.base.core import DataType

    numel = value.numel()
    mapping = {
        DataType.BOOL: 1,
        DataType.INT8: 1,
        DataType.INT16: 2,
        DataType.INT32: 4,
        DataType.INT64: 8,
        DataType.UINT8: 1,
        DataType.UINT16: 2,
        DataType.UINT32: 4,
        DataType.UINT64: 8,
        DataType.FLOAT32: 4,
        DataType.FLOAT64: 8,
    }
    dtype = mapping[value.type().dtype]
    return dtype * numel


def analysis_io(program: paddle.pir.Program):
    # 1. don't support control flow now
    # 2. each op is consider read all inputs and write all outputs once.
    # 3. unit is "GByte"
    total_io = 0.0
    for op in program.global_block().ops:
        for operand in op.operands():
            value = operand.source()
            total_io += get_memory(value)

        for value in op.results():
            total_io += get_memory(value)

    return total_io / 1024 / 1024 / 1024


def append_activation_in_pir(input, act=None, use_cudnn=None):
    if act is None:
        return input

    act_name_mapping = {
        "hard_swish": "hardswish",
    }
    act = act_name_mapping.get(act, act)

    attrs = ()
    if use_cudnn:
        attrs = ('use_cudnn', use_cudnn)
    act_op = getattr(paddle._C_ops, act)
    if act == 'softmax':
        return act_op(input, -1)
    return act_op(input, *attrs)
